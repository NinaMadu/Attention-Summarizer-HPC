/*=============================================================================
 *  Attention Summarizer — CUDA Accelerated Version
 *
 *  GPU-accelerated stages:
 *    - Covariance matrix computation  O(V × D²)
 *    - Multi-head self-attention       O(T × D² + H × T² × d)
 *
 *  CPU-only stages:
 *    - GloVe loading, Jacobi eigendecomposition, tokenization,
 *      vectorization, positional encoding, sentence scoring
 *
 *  Produces identical COMPARE_* output for numerical verification
 *  against the serial baseline (attention_summarizer_new.c).
 *
 *  Build:  nvcc -O2 cuda/attention_summarizer_cuda.cu
 *               -o cuda/summarizer_cuda
 *===========================================================================*/

#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <cuda_runtime.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <time.h>
#endif

/* ====================== CONSTANTS ====================== */

#define MAX_TOKENS     1024
#define MAX_WORD_LEN   64
#define MAX_SENTENCES  128
#define MAX_SENTENCE_LEN 16384
#define MAX_PARAGRAPH_CHARS 65536
#define MAX_SUMMARY_CHARS   65536
#define EMBED_DIM      300
#define MAX_VOCAB      500000
#define NUM_HEADS      12
#define HEAD_DIM       (EMBED_DIM / NUM_HEADS)
#define JACOBI_SWEEPS  100
#define EIG_FLOOR      1e-6f
#define BLOCK_SIZE     256
#define TRANSPOSE_TILE_DIM 32
#define TRANSPOSE_BLOCK_ROWS 8

/* ====================== CUDA ERROR CHECKING ====================== */

#define CUDA_CHECK(call) do {                                        \
    cudaError_t err = (call);                                        \
    if (err != cudaSuccess) {                                        \
        fprintf(stderr, "CUDA ERROR at %s:%d — %s\n",               \
                __FILE__, __LINE__, cudaGetErrorString(err));        \
        exit(EXIT_FAILURE);                                          \
    }                                                                \
} while(0)

/* ====================== CROSS-PLATFORM HIGH-RES TIMER ====================== */

#ifdef _WIN32
static double now_sec() {
    LARGE_INTEGER freq, cnt;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&cnt);
    return (double)cnt.QuadPart / (double)freq.QuadPart;
}
#else
static double now_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
#endif

/* ====================== LOGGING UTILITIES ====================== */

typedef struct {
    const char *name;
    double elapsed_sec;
    long long ops;
} StageLog;

#define MAX_STAGES 16
static StageLog stage_log[MAX_STAGES];
static int stage_count = 0;

static void log_stage(const char *name, double start, double end, long long ops) {
    if (stage_count < MAX_STAGES) {
        stage_log[stage_count].name        = name;
        stage_log[stage_count].elapsed_sec = end - start;
        stage_log[stage_count].ops         = ops;
        stage_count++;
    }
}

static void print_separator(char c, int width) {
    for (int i = 0; i < width; i++) putchar(c);
    putchar('\n');
}

static void print_report(double total_start) {
    double total = now_sec() - total_start;
    printf("\n");
    print_separator('=', 70);
    printf("  COMPLEXITY & TIMING REPORT  [CUDA]\n");
    print_separator('=', 70);
    printf("  %-32s  %10s  %18s  %s\n", "Stage", "Time (s)", "Operations", "% Total");
    print_separator('-', 70);
    for (int i = 0; i < stage_count; i++) {
        double pct = (total > 0) ? 100.0 * stage_log[i].elapsed_sec / total : 0.0;
        int bar_len = (int)(pct / 5.0);
        char bar[22];
        for (int b = 0; b < 20; b++) bar[b] = (b < bar_len) ? '#' : ' ';
        bar[20] = '\0';
        printf("  %-32s  %10.4f  %18lld  %5.1f%% [%s]\n",
               stage_log[i].name,
               stage_log[i].elapsed_sec,
               stage_log[i].ops,
               pct,
               bar);
    }
    print_separator('-', 70);
    printf("  %-32s  %10.4f\n", "TOTAL", total);
    print_separator('=', 70);
    printf("\n  Complexity summary:\n");
    printf("    load_glove           O(V x D)          V=%d, D=%d\n", MAX_VOCAB, EMBED_DIM);
    printf("    covariance_matrix    O(V x D^2)        ~%.1fB multiply-adds  [GPU]\n",
           (double)MAX_VOCAB * EMBED_DIM * EMBED_DIM / 2.0 / 1e9);
    printf("    jacobi_eigen         O(sweeps x D^3)   sweeps=%d => ~%.0fM rotations  [CPU]\n",
           JACOBI_SWEEPS,
           (double)JACOBI_SWEEPS * EMBED_DIM * EMBED_DIM / 2.0 / 1e6);
    printf("    project_embeddings   O(T x D^2)        T=n_tokens  [GPU]\n");
    printf("    self_attention QK^T  O(H x T^2 x d)    H=%d heads, d=%d  [GPU]\n", NUM_HEADS, HEAD_DIM);
    printf("    self_attention out   O(H x T^2 x d)    [GPU]\n");
    print_separator('=', 70);
    printf("\n");
}

/* ====================== JACOBI PROGRESS LOGGING ====================== */
static int jacobi_verbose = 1;

/* ====================== STRUCTS ====================== */

typedef struct {
    char word[MAX_WORD_LEN];
    float vec[EMBED_DIM];
} Embedding;

typedef struct {
    float token_score[MAX_TOKENS];
    float sent_score[MAX_SENTENCES];
    int sent_count[MAX_SENTENCES];
    int selected[2];
} SummaryMetrics;

/* ====================== GLOBALS ====================== */

Embedding *vocab;
int vocab_size = 0;

float W_Q[EMBED_DIM][EMBED_DIM];
float W_K[EMBED_DIM][EMBED_DIM];

/* ===========================================================================
 *                           CUDA KERNELS
 * =========================================================================== */

/*  K1: Parallel mean — one block per embedding dimension.
 *      Each block uses BLOCK_SIZE threads to cooperatively reduce over V
 *      vocabulary vectors using shared-memory parallel reduction.
 *      Grid : EMBED_DIM blocks      Block: BLOCK_SIZE threads              */
__global__ void compute_mean_kernel(const float *vecs, float *mean, int V) {
    int d = blockIdx.x;
    if (d >= EMBED_DIM) return;

    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;

    float local_sum = 0.0f;
    for (int v = tid; v < V; v += blockDim.x)
        local_sum += vecs[(long long)d * V + v];

    sdata[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) mean[d] = sdata[0] / V;
}

/*  K2: Covariance matrix — one block per (i,j) element of cov[D][D].
 *      Threads cooperatively reduce over V vectors (shared-mem reduction).
 *      Only the lower triangle (j <= i) is computed; the result is mirrored.
 *      Grid : (EMBED_DIM, EMBED_DIM)      Block: BLOCK_SIZE                */
__global__ void covariance_kernel(const float *vecs, const float *mean,
                                   float *cov, int V) {
    int pair_idx = (int)blockIdx.x;
    int i = (int)((sqrtf(8.0f * (float)pair_idx + 1.0f) - 1.0f) * 0.5f);
    int row_start = i * (i + 1) / 2;
    if (row_start > pair_idx) {
        i--;
        row_start = i * (i + 1) / 2;
    }
    int j = pair_idx - row_start;
    if (i >= EMBED_DIM || j > i) return;

    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;

    float mi = mean[i], mj = mean[j];
    float local_sum = 0.0f;
    for (int v = tid; v < V; v += blockDim.x)
        local_sum += (vecs[(long long)i * V + v] - mi)
                   * (vecs[(long long)j * V + v] - mj);

    sdata[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) {
        float val = sdata[0] / (V - 1);
        cov[i * EMBED_DIM + j] = val;
        cov[j * EMBED_DIM + i] = val;
    }
}

/*  K3: Project embeddings — matrix multiply   out = in × W
 *      Grid : n_tokens blocks (one block per token)
 *      Block: BLOCK_SIZE threads (threads stride over output dimensions)   */
/*  Convert uploaded vocab vectors from row-major [V][D] to dimension-major [D][V].
 *  Covariance then reads contiguous values across vocabulary rows. */
__global__ void transpose_vocab_kernel(const float *row_major, float *dim_major, int V) {
    __shared__ float tile[TRANSPOSE_TILE_DIM][TRANSPOSE_TILE_DIM + 1];

    int in_d = blockIdx.x * TRANSPOSE_TILE_DIM + threadIdx.x;
    int in_v = blockIdx.y * TRANSPOSE_TILE_DIM + threadIdx.y;

    for (int r = 0; r < TRANSPOSE_TILE_DIM; r += TRANSPOSE_BLOCK_ROWS) {
        int v = in_v + r;
        if (in_d < EMBED_DIM && v < V)
            tile[threadIdx.y + r][threadIdx.x] = row_major[(long long)v * EMBED_DIM + in_d];
    }

    __syncthreads();

    int out_v = blockIdx.y * TRANSPOSE_TILE_DIM + threadIdx.x;
    int out_d = blockIdx.x * TRANSPOSE_TILE_DIM + threadIdx.y;

    for (int r = 0; r < TRANSPOSE_TILE_DIM; r += TRANSPOSE_BLOCK_ROWS) {
        int d = out_d + r;
        if (d < EMBED_DIM && out_v < V)
            dim_major[(long long)d * V + out_v] = tile[threadIdx.x][threadIdx.y + r];
    }
}

__global__ void project_kernel(const float *in, const float *W, float *out,
                                int n_tokens) {
    int i = blockIdx.x;
    if (i >= n_tokens) return;

    for (int j = threadIdx.x; j < EMBED_DIM; j += blockDim.x) {
        float sum = 0.0f;
        for (int k = 0; k < EMBED_DIM; k++)
            sum += in[i * EMBED_DIM + k] * W[k * EMBED_DIM + j];
        out[i * EMBED_DIM + j] = sum;
    }
}

/*  K4: Scaled dot-product  QK^T  for all heads
 *      Grid : (ceil(T/BLOCK), T, H)      Block: BLOCK_SIZE
 *      Each thread computes one score element scores[h][i][j]              */
__global__ void qk_dot_kernel(const float *Q, const float *K, float *scores,
                               int n_tokens, float scale) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y;
    int h = blockIdx.z;
    if (i >= n_tokens || j >= n_tokens) return;

    int offset = h * HEAD_DIM;
    float dot = 0.0f;
    for (int d = 0; d < HEAD_DIM; d++)
        dot += Q[i * EMBED_DIM + offset + d] * K[j * EMBED_DIM + offset + d];
    scores[(long long)h * MAX_TOKENS * MAX_TOKENS + i * MAX_TOKENS + j] = dot * scale;
}

/*  K5: Row-wise stable softmax (subtract max before exp)
 *      Each thread processes one row (head h, query i).
 *      Grid : ceil(H*T / BLOCK)      Block: BLOCK_SIZE                    */
__global__ void softmax_kernel(float *scores, int n_heads, int n_tokens) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int h = idx / n_tokens;
    int i = idx % n_tokens;
    if (h >= n_heads) return;

    float *row = scores + (long long)h * MAX_TOKENS * MAX_TOKENS + i * MAX_TOKENS;

    float mx = row[0];
    for (int j = 1; j < n_tokens; j++)
        if (row[j] > mx) mx = row[j];

    float sum = 0.0f;
    for (int j = 0; j < n_tokens; j++) {
        row[j] = expf(row[j] - mx);
        sum += row[j];
    }

    for (int j = 0; j < n_tokens; j++)
        row[j] /= sum;
}

/*  K6: Attention-weighted output
 *      out[i][offset+dd] = Σ_j  attn[h][i][j] × V[j][offset+dd]
 *      Grid : (ceil(HEAD_DIM/32), T, H)      Block: 32                    */
__global__ void attn_output_kernel(const float *attn, const float *V_mat,
                                    float *out, int n_tokens) {
    int dd = blockIdx.x * blockDim.x + threadIdx.x;
    int i  = blockIdx.y;
    int h  = blockIdx.z;
    if (dd >= HEAD_DIM || i >= n_tokens) return;

    int offset = h * HEAD_DIM;
    float sum = 0.0f;
    for (int j = 0; j < n_tokens; j++)
        sum += attn[(long long)h * MAX_TOKENS * MAX_TOKENS + i * MAX_TOKENS + j]
             * V_mat[j * EMBED_DIM + offset + dd];
    out[i * EMBED_DIM + offset + dd] = sum;
}

/*  K7: Average attention weights across all heads
 *      Grid : (ceil(T/BLOCK), T)      Block: BLOCK_SIZE                   */
__global__ void average_heads_kernel(const float *head_attn, float *avg_attn,
                                      int n_tokens) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y;
    if (i >= n_tokens || j >= n_tokens) return;

    float sum = 0.0f;
    for (int h = 0; h < NUM_HEADS; h++)
        sum += head_attn[(long long)h * MAX_TOKENS * MAX_TOKENS + i * MAX_TOKENS + j];
    avg_attn[i * MAX_TOKENS + j] = sum / NUM_HEADS;
}

/*  K8: Per-head average diagonal attention for diagnostics.
 *      Keeps the existing printed metric without copying H x T x T scores. */
__global__ void head_diag_mean_kernel(const float *head_attn, float *diag_mean,
                                      int n_tokens) {
    int h = blockIdx.x;
    int tid = threadIdx.x;
    if (h >= NUM_HEADS) return;

    __shared__ float sdata[BLOCK_SIZE];
    float local_sum = 0.0f;
    for (int i = tid; i < n_tokens; i += blockDim.x)
        local_sum += head_attn[(long long)h * MAX_TOKENS * MAX_TOKENS
                               + i * MAX_TOKENS + i];

    sdata[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) diag_mean[h] = sdata[0] / n_tokens;
}

/* ===========================================================================
 *                     CPU FUNCTIONS  (identical to serial)
 * =========================================================================== */

/* ====================== JACOBI EIGENDECOMPOSITION ====================== */
void jacobi_eigen(float A[EMBED_DIM][EMBED_DIM],
                  float V[EMBED_DIM][EMBED_DIM],
                  float d[EMBED_DIM],
                  long long *op_count) {
    int n = EMBED_DIM;
    *op_count = 0;

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            V[i][j] = (i == j) ? 1.0f : 0.0f;

    float S[EMBED_DIM][EMBED_DIM];
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            S[i][j] = A[i][j];

    printf("  [jacobi] Starting %d sweeps on %dx%d matrix...\n",
           JACOBI_SWEEPS, EMBED_DIM, EMBED_DIM);

    for (int sweep = 0; sweep < JACOBI_SWEEPS; sweep++) {
        float off = 0.0f;
        for (int i = 0; i < n; i++)
            for (int j = i+1; j < n; j++)
                off += S[i][j] * S[i][j];

        if (jacobi_verbose && (sweep % 10 == 0 || sweep == JACOBI_SWEEPS-1)) {
            printf("  [jacobi] sweep %3d/%d  off-diag norm = %.6e  "
                   "rotations so far = %lld\n",
                   sweep, JACOBI_SWEEPS, sqrtf(off), *op_count);
        }
        if (off < 1e-10f) {
            printf("  [jacobi] Converged at sweep %d (off < 1e-10)\n", sweep);
            break;
        }

        for (int p = 0; p < n-1; p++) {
            for (int q = p+1; q < n; q++) {
                if (fabsf(S[p][q]) < 1e-12f) continue;

                float theta = 0.5f * atan2f(2.0f * S[p][q], S[p][p] - S[q][q]);
                float c = cosf(theta);
                float s = sinf(theta);

                float new_Spp = c*c*S[p][p] + 2*s*c*S[p][q] + s*s*S[q][q];
                float new_Sqq = s*s*S[p][p] - 2*s*c*S[p][q] + c*c*S[q][q];

                for (int r = 0; r < n; r++) {
                    if (r == p || r == q) continue;
                    float new_rp = c*S[r][p] + s*S[r][q];
                    float new_rq = -s*S[r][p] + c*S[r][q];
                    S[r][p] = S[p][r] = new_rp;
                    S[r][q] = S[q][r] = new_rq;
                    *op_count += 4;
                }
                S[p][p] = new_Spp;
                S[q][q] = new_Sqq;
                S[p][q] = S[q][p] = 0.0f;

                for (int r = 0; r < n; r++) {
                    float new_rp =  c*V[r][p] + s*V[r][q];
                    float new_rq = -s*V[r][p] + c*V[r][q];
                    V[r][p] = new_rp;
                    V[r][q] = new_rq;
                    *op_count += 4;
                }
            }
        }
    }

    for (int i = 0; i < n; i++)
        d[i] = S[i][i];

    printf("  [jacobi] Done. Total floating-point ops: %lld\n", *op_count);
}

/* ====================== PCA PROJECTIONS (GPU covariance) ====================== */
void compute_pca_projections_cuda(double total_start) {
    /* ---- Stage: covariance_matrix [GPU] ---- */
    printf("\n");
    print_separator('-', 60);
    printf("  STAGE: covariance_matrix  [CUDA]\n");
    printf("  Complexity: O(V x D^2)  =>  V=%d, D=%d\n", vocab_size, EMBED_DIM);
    printf("  Expected ops: ~%.2fB\n",
           (double)vocab_size * EMBED_DIM * EMBED_DIM / 2.0 / 1e9);
    print_separator('-', 60);

    double t0 = now_sec();

    /* Extract float vectors from AoS vocab[] into row-major pinned upload memory. */
    size_t vecs_bytes = (size_t)vocab_size * EMBED_DIM * sizeof(float);
    float *flat_vecs = NULL;
    int flat_vecs_pinned = 1;
    cudaError_t host_err = cudaMallocHost((void **)&flat_vecs, vecs_bytes);
    if (host_err != cudaSuccess) {
        flat_vecs_pinned = 0;
        flat_vecs = (float *)malloc(vecs_bytes);
        if (!flat_vecs) {
            printf("ERROR: flat_vecs alloc failed\n");
            return;
        }
        printf("  [cov] Pinned host allocation unavailable (%s); using pageable memory.\n",
               cudaGetErrorString(host_err));
        cudaGetLastError();
    }
    for (int v = 0; v < vocab_size; v++)
        memcpy(flat_vecs + (size_t)v * EMBED_DIM, vocab[v].vec, EMBED_DIM * sizeof(float));

    float *d_vecs_row, *d_vecs, *d_mean, *d_cov;
    CUDA_CHECK(cudaMalloc(&d_vecs_row, vecs_bytes));
    CUDA_CHECK(cudaMalloc(&d_vecs, vecs_bytes));
    CUDA_CHECK(cudaMalloc(&d_mean, EMBED_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cov, EMBED_DIM * EMBED_DIM * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_vecs_row, flat_vecs, vecs_bytes, cudaMemcpyHostToDevice));
    if (flat_vecs_pinned) cudaFreeHost(flat_vecs);
    else free(flat_vecs);

    /* Convert once on GPU so mean/covariance read contiguous vocabulary values. */
    dim3 tr_block(TRANSPOSE_TILE_DIM, TRANSPOSE_BLOCK_ROWS);
    dim3 tr_grid((EMBED_DIM + TRANSPOSE_TILE_DIM - 1) / TRANSPOSE_TILE_DIM,
                 (vocab_size + TRANSPOSE_TILE_DIM - 1) / TRANSPOSE_TILE_DIM);
    transpose_vocab_kernel<<<tr_grid, tr_block>>>(d_vecs_row, d_vecs, vocab_size);
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaFree(d_vecs_row);

    printf("  [cov] Computing mean vector on GPU (%d blocks x %d threads)...\n",
           EMBED_DIM, BLOCK_SIZE);
    compute_mean_kernel<<<EMBED_DIM, BLOCK_SIZE>>>(d_vecs, d_mean, vocab_size);
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int cov_pairs = EMBED_DIM * (EMBED_DIM + 1) / 2;
    dim3 cov_grid(cov_pairs);
    printf("  [cov] Computing %dx%d covariance matrix on GPU (%d lower-triangle blocks x %d threads)...\n",
           EMBED_DIM, EMBED_DIM, cov_pairs, BLOCK_SIZE);
    covariance_kernel<<<cov_grid, BLOCK_SIZE>>>(d_vecs, d_mean, d_cov, vocab_size);
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float *cov_flat = NULL;
    int cov_flat_pinned = 1;
    host_err = cudaMallocHost((void **)&cov_flat, EMBED_DIM * EMBED_DIM * sizeof(float));
    if (host_err != cudaSuccess) {
        cov_flat_pinned = 0;
        cov_flat = (float *)malloc(EMBED_DIM * EMBED_DIM * sizeof(float));
        if (!cov_flat) {
            printf("ERROR: cov_flat alloc failed\n");
            cudaFree(d_vecs);
            cudaFree(d_mean);
            cudaFree(d_cov);
            return;
        }
        printf("  [cov] Pinned covariance download allocation unavailable (%s); using pageable memory.\n",
               cudaGetErrorString(host_err));
        cudaGetLastError();
    }
    CUDA_CHECK(cudaMemcpy(cov_flat, d_cov, EMBED_DIM * EMBED_DIM * sizeof(float),
                           cudaMemcpyDeviceToHost));

    /* Free large GPU allocations (vocab vectors ~400MB) */
    cudaFree(d_vecs);
    cudaFree(d_mean);
    cudaFree(d_cov);

    double t1 = now_sec();
    long long cov_ops = (long long)vocab_size * EMBED_DIM * (EMBED_DIM + 1) / 2;
    log_stage("covariance_matrix", t0, t1, cov_ops);
    printf("  [cov] Finished in %.4fs  total ops: %lld\n", t1 - t0, cov_ops);

    /* ---- Stage: jacobi_eigen [CPU] ---- */
    printf("\n");
    print_separator('-', 60);
    printf("  STAGE: jacobi_eigen  [CPU]\n");
    printf("  Complexity: O(sweeps x D^3)  =>  ~%.0fM Jacobi rotations\n",
           (double)JACOBI_SWEEPS * EMBED_DIM * EMBED_DIM / 2.0 / 1e6);
    print_separator('-', 60);

    double t2 = now_sec();
    float (*cov)[EMBED_DIM] = (float (*)[EMBED_DIM])cov_flat;
    float (*eigvecs)[EMBED_DIM] = (float (*)[EMBED_DIM])malloc(EMBED_DIM * sizeof(*eigvecs));
    float eigvals[EMBED_DIM];
    if (!eigvecs) {
        printf("ERROR: eigvec alloc failed\n");
        if (cov_flat_pinned) cudaFreeHost(cov_flat);
        else free(cov_flat);
        return;
    }

    long long jacobi_ops = 0;
    jacobi_eigen(cov, eigvecs, eigvals, &jacobi_ops);

    double t3 = now_sec();
    log_stage("jacobi_eigen", t2, t3, jacobi_ops);
    printf("  [jacobi] Finished in %.4fs\n", t3 - t2);

    /* ---- Stage: build_whitening_matrices [CPU] ---- */
    printf("\n");
    print_separator('-', 60);
    printf("  STAGE: build_whitening_matrices  [CPU]\n");
    printf("  Complexity: O(D^2) = %d ops\n", EMBED_DIM * EMBED_DIM);
    print_separator('-', 60);

    double t4 = now_sec();
    long long w_ops = 0;

    printf("  [pca] Top 5 eigenvalues (variance explained):\n");
    float ev_copy[EMBED_DIM];
    memcpy(ev_copy, eigvals, sizeof(eigvals));
    for (int pass = 0; pass < 5; pass++) {
        int mx = pass;
        for (int k = pass + 1; k < EMBED_DIM; k++)
            if (ev_copy[k] > ev_copy[mx]) mx = k;
        float tmp = ev_copy[pass]; ev_copy[pass] = ev_copy[mx]; ev_copy[mx] = tmp;
        printf("    eigenvalue[%d] = %.6f\n", pass, ev_copy[pass]);
    }

    for (int i = 0; i < EMBED_DIM; i++) {
        for (int j = 0; j < EMBED_DIM; j++) {
            float scale = (eigvals[j] > EIG_FLOOR) ? 1.0f / sqrtf(eigvals[j]) : 0.0f;
            W_Q[i][j] = eigvecs[i][j] * scale;
            W_K[i][j] = eigvecs[i][j] * scale;
            w_ops++;
        }
    }
    double t5 = now_sec();
    log_stage("build_whitening_W", t4, t5, w_ops);
    printf("  [pca] W_Q / W_K built in %.6fs\n", t5 - t4);

    if (cov_flat_pinned) cudaFreeHost(cov_flat);
    else free(cov_flat);
    free(eigvecs);
}

/* ====================== LOAD GLOVE ====================== */
int load_glove(const char *filename) {
    printf("\n");
    print_separator('-', 60);
    printf("  STAGE: load_glove\n");
    printf("  Complexity: O(V x D) reads, V<=%d, D=%d\n", MAX_VOCAB, EMBED_DIM);
    print_separator('-', 60);

    double t0 = now_sec();

    FILE *f = fopen(filename, "r");
    if (!f) { printf("ERROR: Cannot open %s\n", filename); return 0; }

    vocab = (Embedding *)malloc(MAX_VOCAB * sizeof(Embedding));
    if (!vocab) return 0;

    char line[16384];
    vocab_size = 0;
    while (fgets(line, sizeof(line), f) && vocab_size < MAX_VOCAB) {
        char *token = strtok(line, " ");
        if (!token) continue;
        strncpy(vocab[vocab_size].word, token, MAX_WORD_LEN - 1);
        vocab[vocab_size].word[MAX_WORD_LEN - 1] = '\0';
        for (int dd = 0; dd < EMBED_DIM; dd++) {
            token = strtok(NULL, " ");
            if (token) vocab[vocab_size].vec[dd] = (float)atof(token);
        }
        vocab_size++;
        if (vocab_size % 100000 == 0)
            printf("  [load] Loaded %d vectors so far...\n", vocab_size);
    }
    fclose(f);

    double t1 = now_sec();
    long long load_ops = (long long)vocab_size * EMBED_DIM;
    log_stage("load_glove", t0, t1, load_ops);
    printf("  [load] Loaded %d GloVe vectors (%dD) in %.4fs\n",
           vocab_size, EMBED_DIM, t1 - t0);
    return 1;
}

/* ====================== TOKENIZE ====================== */
static int is_token_delim(char c) {
    return c == '\0' || strchr(" .,;:!?\n\t\"'()[]{}", c) != NULL;
}

static int is_sentence_delim(char c) {
    return c == '.' || c == '!' || c == '?';
}

static void trim_sentence(char *s) {
    int start = 0;
    while (s[start] && isspace((unsigned char)s[start])) start++;
    if (start > 0) memmove(s, s + start, strlen(s + start) + 1);

    int end = (int)strlen(s) - 1;
    while (end >= 0 && isspace((unsigned char)s[end])) s[end--] = '\0';
}

int tokenize(char *text,
             char tokens[MAX_TOKENS][MAX_WORD_LEN],
             int token_sentence[MAX_TOKENS],
             char sentences[MAX_SENTENCES][MAX_SENTENCE_LEN],
             int *n_sentences) {
    int n = 0;
    int s = 0;
    int sent_len = 0;
    int tok_len = 0;
    char token_buf[MAX_WORD_LEN];

    memset(token_sentence, -1, sizeof(int) * MAX_TOKENS);
    for (int i = 0; i < MAX_SENTENCES; i++) sentences[i][0] = '\0';

    for (int i = 0; ; i++) {
        char c = text[i];

        if (c != '\0' && !is_sentence_delim(c) && s < MAX_SENTENCES) {
            if (sent_len < MAX_SENTENCE_LEN - 1) {
                sentences[s][sent_len++] = c;
                sentences[s][sent_len] = '\0';
            }
        }

        if (!is_token_delim(c)) {
            if (tok_len < MAX_WORD_LEN - 1)
                token_buf[tok_len++] = (char)tolower((unsigned char)c);
        } else if (tok_len > 0 && n < MAX_TOKENS) {
            token_buf[tok_len] = '\0';
            strncpy(tokens[n], token_buf, MAX_WORD_LEN - 1);
            tokens[n][MAX_WORD_LEN - 1] = '\0';
            token_sentence[n] = s;
            n++;
            tok_len = 0;
        } else {
            tok_len = 0;
        }

        if (is_sentence_delim(c) || c == '\0') {
            if (s < MAX_SENTENCES && sent_len > 0) {
                trim_sentence(sentences[s]);
                if (strlen(sentences[s]) > 0) s++;
            }
            sent_len = 0;
            if (s >= MAX_SENTENCES || c == '\0') break;
        }

        if (n >= MAX_TOKENS || c == '\0') break;
    }

    *n_sentences = s;
    return n;
}

/* ====================== VECTORIZE ====================== */
void vectorize(char tokens[MAX_TOKENS][MAX_WORD_LEN], int n_tokens,
               float embeddings[MAX_TOKENS][EMBED_DIM]) {
    printf("\n");
    print_separator('-', 60);
    printf("  STAGE: vectorize\n");
    printf("  Complexity: O(T x V) linear scan,  T=%d, V=%d\n", n_tokens, vocab_size);
    printf("  Worst-case comparisons: %lld\n", (long long)n_tokens * vocab_size);
    print_separator('-', 60);

    double t0 = now_sec();
    int hit = 0;
    long long cmp_ops = 0;
    for (int i = 0; i < n_tokens; i++) {
        int found = 0;
        for (int v = 0; v < vocab_size; v++) {
            cmp_ops++;
            if (strcmp(tokens[i], vocab[v].word) == 0) {
                memcpy(embeddings[i], vocab[v].vec, sizeof(float) * EMBED_DIM);
                found = 1; hit++;
                break;
            }
        }
        if (!found) {
            memset(embeddings[i], 0, sizeof(float) * EMBED_DIM);
            printf("  [oov] '%s' not in GloVe vocab\n", tokens[i]);
        }
    }
    double t1 = now_sec();
    log_stage("tokenize+vectorize", t0, t1, cmp_ops);
    printf("  [vec] Coverage: %d/%d tokens hit  comparisons: %lld  time: %.4fs\n",
           hit, n_tokens, cmp_ops, t1 - t0);
}

/* ====================== POSITIONAL ENCODING ====================== */
void add_positional_encoding(float embeddings[MAX_TOKENS][EMBED_DIM], int n_tokens) {
    printf("\n");
    print_separator('-', 60);
    printf("  STAGE: positional_encoding\n");
    printf("  Complexity: O(T x D)  ops: %d\n", n_tokens * EMBED_DIM);
    print_separator('-', 60);

    double t0 = now_sec();
    long long ops = 0;
    for (int pos = 0; pos < n_tokens; pos++) {
        for (int dd = 0; dd < EMBED_DIM; dd++) {
            float angle = pos / powf(10000.0f, (2.0f * (dd / 2)) / (float)EMBED_DIM);
            embeddings[pos][dd] += (dd % 2 == 0) ? sinf(angle) : cosf(angle);
            ops++;
        }
    }
    double t1 = now_sec();
    log_stage("positional_encoding", t0, t1, ops);
    printf("  [pe] Done in %.6fs  ops: %lld\n", t1 - t0, ops);
}

/* ====================== MULTI-HEAD SELF-ATTENTION [CUDA] ====================== */
void multihead_self_attention_cuda(float embeddings[MAX_TOKENS][EMBED_DIM],
                                    int n_tokens,
                                    float attn_weights[MAX_TOKENS][MAX_TOKENS],
                                    float output[MAX_TOKENS][EMBED_DIM]) {
    printf("\n");
    print_separator('-', 60);
    printf("  STAGE: multi_head_self_attention  [CUDA]\n");
    printf("  T=%d  H=%d heads  head_dim d=%d\n", n_tokens, NUM_HEADS, HEAD_DIM);
    printf("  Project Q,K:  O(T x D^2) = %lld ops each\n",
           (long long)n_tokens * EMBED_DIM * EMBED_DIM);
    printf("  QK^T:         O(H x T^2 x d) = %lld ops\n",
           (long long)NUM_HEADS * n_tokens * n_tokens * HEAD_DIM);
    printf("  Softmax:      O(H x T^2) = %d ops\n", NUM_HEADS * n_tokens * n_tokens);
    printf("  Output V:     O(H x T^2 x d) = %lld ops\n",
           (long long)NUM_HEADS * n_tokens * n_tokens * HEAD_DIM);
    print_separator('-', 60);

    double t0 = now_sec();

    float *d_embeddings, *d_W_Q, *d_W_K, *d_Q, *d_K;
    float *d_head_scores, *d_output, *d_attn_weights;

    size_t embed_bytes  = MAX_TOKENS * EMBED_DIM * sizeof(float);
    size_t W_bytes      = EMBED_DIM * EMBED_DIM * sizeof(float);
    size_t scores_bytes = (size_t)NUM_HEADS * MAX_TOKENS * MAX_TOKENS * sizeof(float);
    size_t attn_bytes   = (size_t)MAX_TOKENS * MAX_TOKENS * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_embeddings,  embed_bytes));
    CUDA_CHECK(cudaMalloc(&d_W_Q,         W_bytes));
    CUDA_CHECK(cudaMalloc(&d_W_K,         W_bytes));
    CUDA_CHECK(cudaMalloc(&d_Q,           embed_bytes));
    CUDA_CHECK(cudaMalloc(&d_K,           embed_bytes));
    CUDA_CHECK(cudaMalloc(&d_head_scores, scores_bytes));
    CUDA_CHECK(cudaMalloc(&d_output,      embed_bytes));
    CUDA_CHECK(cudaMalloc(&d_attn_weights, attn_bytes));

    CUDA_CHECK(cudaMemset(d_head_scores,  0, scores_bytes));
    CUDA_CHECK(cudaMemset(d_output,       0, embed_bytes));
    CUDA_CHECK(cudaMemset(d_attn_weights, 0, attn_bytes));

    CUDA_CHECK(cudaMemcpy(d_embeddings, embeddings, embed_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W_Q, W_Q, W_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W_K, W_K, W_bytes, cudaMemcpyHostToDevice));

    printf("  [attn] Projecting Q (GPU)...\n");
    double tp0 = now_sec();
    project_kernel<<<n_tokens, BLOCK_SIZE>>>(d_embeddings, d_W_Q, d_Q, n_tokens);
    CUDA_CHECK(cudaDeviceSynchronize());
    double tp1 = now_sec();
    long long proj_ops = (long long)n_tokens * EMBED_DIM * EMBED_DIM;
    printf("  [attn] Q projected in %.4fs  (%lld multiply-adds)\n", tp1 - tp0, proj_ops);

    printf("  [attn] Projecting K (GPU)...\n");
    project_kernel<<<n_tokens, BLOCK_SIZE>>>(d_embeddings, d_W_K, d_K, n_tokens);
    CUDA_CHECK(cudaDeviceSynchronize());
    double tp2 = now_sec();
    long long k_ops = (long long)n_tokens * EMBED_DIM * EMBED_DIM;
    printf("  [attn] K projected in %.4fs  (%lld multiply-adds)\n", tp2 - tp1, k_ops);
    proj_ops += k_ops;

    float scale = 1.0f / sqrtf((float)HEAD_DIM);
    printf("  [attn] Attention scale factor: 1/sqrt(%d) = %.6f\n", HEAD_DIM, scale);

    dim3 qk_grid((n_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE, n_tokens, NUM_HEADS);
    dim3 qk_block(BLOCK_SIZE);
    qk_dot_kernel<<<qk_grid, qk_block>>>(d_Q, d_K, d_head_scores, n_tokens, scale);
    CUDA_CHECK(cudaDeviceSynchronize());

    int total_rows = NUM_HEADS * n_tokens;
    int sm_grid = (total_rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
    softmax_kernel<<<sm_grid, BLOCK_SIZE>>>(d_head_scores, NUM_HEADS, n_tokens);
    CUDA_CHECK(cudaDeviceSynchronize());

    /* ---- Per-head diagnostics: reduce on GPU, copy only NUM_HEADS floats ---- */
    float *d_diag_mean = NULL;
    float h_diag_mean[NUM_HEADS];
    CUDA_CHECK(cudaMalloc(&d_diag_mean, NUM_HEADS * sizeof(float)));
    head_diag_mean_kernel<<<NUM_HEADS, BLOCK_SIZE>>>(d_head_scores, d_diag_mean, n_tokens);
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaMemcpy(h_diag_mean, d_diag_mean, NUM_HEADS * sizeof(float),
                          cudaMemcpyDeviceToHost));
    cudaFree(d_diag_mean);
    long long qk_ops = 0;
    for (int h = 0; h < NUM_HEADS; h++) {
        printf("  [attn] head %d/%d  avg self-attention weight = %.4f\n",
               h + 1, NUM_HEADS, h_diag_mean[h]);
    }
    qk_ops = (long long)NUM_HEADS * n_tokens * n_tokens * HEAD_DIM;

    dim3 out_grid((HEAD_DIM + 31) / 32, n_tokens, NUM_HEADS);
    dim3 out_block(32);
    attn_output_kernel<<<out_grid, out_block>>>(d_head_scores, d_embeddings,
                                                 d_output, n_tokens);
    CUDA_CHECK(cudaDeviceSynchronize());

    dim3 avg_grid((n_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE, n_tokens);
    dim3 avg_block(BLOCK_SIZE);
    average_heads_kernel<<<avg_grid, avg_block>>>(d_head_scores, d_attn_weights,
                                                    n_tokens);
    CUDA_CHECK(cudaDeviceSynchronize());

    memset(output,       0, MAX_TOKENS * EMBED_DIM   * sizeof(float));
    memset(attn_weights, 0, MAX_TOKENS * MAX_TOKENS  * sizeof(float));
    CUDA_CHECK(cudaMemcpy(output,       d_output,       embed_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(attn_weights, d_attn_weights, attn_bytes,  cudaMemcpyDeviceToHost));

    double t1 = now_sec();
    long long out_ops = (long long)NUM_HEADS * n_tokens * HEAD_DIM * n_tokens;
    long long total_attn_ops = proj_ops + qk_ops + out_ops;
    log_stage("self_attention", t0, t1, total_attn_ops);
    printf("  [attn] Total ops — project:%lld  QKt:%lld  output:%lld  sum:%lld\n",
           proj_ops, qk_ops, out_ops, total_attn_ops);
    printf("  [attn] Attention stage finished in %.4fs\n", t1 - t0);

    cudaFree(d_embeddings);
    cudaFree(d_W_Q);
    cudaFree(d_W_K);
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_head_scores);
    cudaFree(d_output);
    cudaFree(d_attn_weights);
}

/* ====================== TOKEN IMPORTANCE ====================== */
float token_importance(float *output_vec) {
    float norm = 0.0f;
    for (int dd = 0; dd < EMBED_DIM; dd++)
        norm += output_vec[dd] * output_vec[dd];
    return sqrtf(norm);
}

/* ====================== SUMMARIZE ====================== */
void summarize(char tokens[MAX_TOKENS][MAX_WORD_LEN],
               int token_sentence[MAX_TOKENS],
               char sentences[MAX_SENTENCES][MAX_SENTENCE_LEN],
               int n_sent,
               int n_tokens,
               float attn_weights[MAX_TOKENS][MAX_TOKENS],
               float output[MAX_TOKENS][EMBED_DIM],
               SummaryMetrics *metrics,
               char *summary) {
    printf("\n");
    print_separator('-', 60);
    printf("  STAGE: sentence_scoring\n");
    printf("  Complexity: O(T^2 + T) attention-centrality scoring, T=%d tokens\n", n_tokens);
    print_separator('-', 60);

    double t0 = now_sec();
    (void)tokens;
    (void)output;
    memset(metrics, 0, sizeof(*metrics));
    metrics->selected[0] = -1;
    metrics->selected[1] = -1;

    printf("  [sum] Detected %d sentences\n", n_sent);

    for (int j = 0; j < n_tokens; j++) {
        for (int i = 0; i < n_tokens; i++)
            metrics->token_score[j] += attn_weights[i][j];
        metrics->token_score[j] /= (n_tokens > 0) ? n_tokens : 1;
    }

    for (int i = 0; i < n_tokens; i++) {
        int s = token_sentence[i];
        if (s < 0 || s >= n_sent) continue;
        metrics->sent_score[s] += metrics->token_score[i];
        metrics->sent_count[s]++;
    }
    printf("  [sum] Sentence scores (avg incoming attention centrality):\n");
    for (int s = 0; s < n_sent; s++) {
        if (metrics->sent_count[s] > 0) {
            metrics->sent_score[s] /= metrics->sent_count[s];
            printf("    sentence[%2d]  tokens=%2d  score=%.4f  preview: \"%.50s...\"\n",
                   s, metrics->sent_count[s], metrics->sent_score[s], sentences[s]);
        }
    }

    int best[2] = {-1, -1};
    float bscore[2] = {-1.0f, -1.0f};
    for (int s = 0; s < n_sent; s++) {
        if (metrics->sent_score[s] > bscore[0])      { bscore[1]=bscore[0]; best[1]=best[0]; bscore[0]=metrics->sent_score[s]; best[0]=s; }
        else if (metrics->sent_score[s] > bscore[1]) { bscore[1]=metrics->sent_score[s]; best[1]=s; }
    }
    metrics->selected[0] = best[0];
    metrics->selected[1] = best[1];
    printf("  [sum] Selected sentences: [%d] (score=%.4f), [%d] (score=%.4f)\n",
           best[0], bscore[0], best[1], bscore[1]);

    int order[2];
    if (best[0] != -1 && best[1] != -1)
        order[0] = (best[0] < best[1]) ? best[0] : best[1],
        order[1] = (best[0] < best[1]) ? best[1] : best[0];
    else order[0] = best[0], order[1] = best[1];

    double t1 = now_sec();
    long long sum_ops = (long long)n_tokens * n_tokens + n_tokens;
    log_stage("sentence_scoring", t0, t1, sum_ops);
    printf("  [sum] Finished in %.6fs\n", t1 - t0);

    summary[0] = '\0';
    strcat(summary, "=== SUMMARY (GloVe 300D + PCA Whitening Attention Centrality) ===\n");
    for (int k = 0; k < 2; k++)
        if (order[k] != -1)
            { strcat(summary, sentences[order[k]]); strcat(summary, ".\n"); }
    strcat(summary, "======================================================\n");
}

/* ====================== COMPARISON OUTPUT ====================== */
void print_comparison_report(char tokens[MAX_TOKENS][MAX_WORD_LEN],
                             int token_sentence[MAX_TOKENS],
                             int n_tokens,
                             int n_sent,
                             float attn_weights[MAX_TOKENS][MAX_TOKENS],
                             float output[MAX_TOKENS][EMBED_DIM],
                             SummaryMetrics *metrics) {
    double attn_checksum = 0.0;
    double attn_abs_checksum = 0.0;
    double out_checksum = 0.0;
    double out_abs_checksum = 0.0;
    double row_sum_min = 1e30;
    double row_sum_max = -1e30;
    float attn_min = 1e30f;
    float attn_max = -1e30f;

    for (int i = 0; i < n_tokens; i++) {
        double row_sum = 0.0;
        for (int j = 0; j < n_tokens; j++) {
            float v = attn_weights[i][j];
            double weight = (double)(i + 1) * (double)(j + 1);
            attn_checksum += (double)v * weight;
            attn_abs_checksum += fabs((double)v) * weight;
            row_sum += v;
            if (v < attn_min) attn_min = v;
            if (v > attn_max) attn_max = v;
        }
        if (row_sum < row_sum_min) row_sum_min = row_sum;
        if (row_sum > row_sum_max) row_sum_max = row_sum;
    }

    for (int i = 0; i < n_tokens; i++) {
        for (int dd = 0; dd < EMBED_DIM; dd++) {
            float v = output[i][dd];
            double weight = (double)(i + 1) * (double)(dd + 1);
            out_checksum += (double)v * weight;
            out_abs_checksum += fabs((double)v) * weight;
        }
    }
    if (n_tokens == 0) {
        row_sum_min = 0.0;
        row_sum_max = 0.0;
        attn_min = 0.0f;
        attn_max = 0.0f;
    }

    printf("\n");
    print_separator('=', 70);
    printf("  COMPARISON REPORT FOR SERIAL / MPI / OPENMP / CUDA\n");
    print_separator('=', 70);
    printf("COMPARE_CONFIG,EMBED_DIM,%d,NUM_HEADS,%d,HEAD_DIM,%d,JACOBI_SWEEPS,%d\n",
           EMBED_DIM, NUM_HEADS, HEAD_DIM, JACOBI_SWEEPS);
    printf("COMPARE_COUNTS,tokens,%d,sentences,%d,vocab,%d\n",
           n_tokens, n_sent, vocab_size);
    printf("COMPARE_SELECTED,rank,1,sentence_id,%d,score,%.9f\n",
           metrics->selected[0],
           (metrics->selected[0] >= 0) ? metrics->sent_score[metrics->selected[0]] : -1.0f);
    printf("COMPARE_SELECTED,rank,2,sentence_id,%d,score,%.9f\n",
           metrics->selected[1],
           (metrics->selected[1] >= 0) ? metrics->sent_score[metrics->selected[1]] : -1.0f);
    printf("COMPARE_ATTENTION,checksum,%.9e,abs_checksum,%.9e,min,%.9e,max,%.9e,row_sum_min,%.9e,row_sum_max,%.9e\n",
           attn_checksum, attn_abs_checksum, (double)attn_min, (double)attn_max, row_sum_min, row_sum_max);
    printf("COMPARE_OUTPUT,checksum,%.9e,abs_checksum,%.9e\n",
           out_checksum, out_abs_checksum);

    printf("\nCOMPARE_SENTENCE_TABLE\n");
    printf("sentence_id,token_count,score\n");
    for (int s = 0; s < n_sent; s++)
        printf("%d,%d,%.9f\n", s, metrics->sent_count[s], metrics->sent_score[s]);

    printf("\nCOMPARE_TOKEN_TABLE\n");
    printf("token_id,sentence_id,token,attention_centrality,output_l2_norm\n");
    for (int i = 0; i < n_tokens; i++)
        printf("%d,%d,%s,%.9f,%.9f\n",
               i, token_sentence[i], tokens[i],
               metrics->token_score[i], token_importance(output[i]));

    int sample = (n_tokens < 8) ? n_tokens : 8;
    printf("\nCOMPARE_ATTENTION_SAMPLE_%dx%d\n", sample, sample);
    for (int si = 0; si < sample; si++) {
        int i = (sample <= 1) ? 0 : (int)((long long)si * (n_tokens - 1) / (sample - 1));
        for (int sj = 0; sj < sample; sj++) {
            int j = (sample <= 1) ? 0 : (int)((long long)sj * (n_tokens - 1) / (sample - 1));
            printf("%s%.9f", (sj == 0) ? "" : ",", attn_weights[i][j]);
        }
        printf("\n");
    }
    print_separator('=', 70);
}

/* ====================== TOP-10 IMPORTANT WORDS ====================== */

static const char *STOPWORDS[] = {
    "the","a","an","is","are","was","were","be","been","being",
    "have","has","had","do","does","did","will","would","could",
    "should","may","might","shall","can","to","of","in","for",
    "on","with","at","by","from","as","into","through","and",
    "or","but","if","then","that","this","it","its","i","we",
    "you","he","she","they","their","our","my","your","his","her",
    "not","no","so","up","out","about","which","who","what","when",
    "where","how","all","each","both","few","more","most","other",
    "some","such","than","too","very","just","also","there","here",
    NULL
};

static int is_stopword(const char *w) {
    for (int i = 0; STOPWORDS[i]; i++)
        if (strcmp(w, STOPWORDS[i]) == 0) return 1;
    return 0;
}

void print_top_words(char tokens[MAX_TOKENS][MAX_WORD_LEN],
                     int n_tokens,
                     float output[MAX_TOKENS][EMBED_DIM]) {
#define TOP_N 10
    typedef struct { char word[MAX_WORD_LEN]; float score; } WordScore;
    WordScore *ws = (WordScore *)malloc(n_tokens * sizeof(WordScore));
    if (!ws) return;
    int ws_count = 0;

    for (int i = 0; i < n_tokens; i++) {
        if (is_stopword(tokens[i])) continue;
        if (strlen(tokens[i]) < 2)  continue;

        float score = token_importance(output[i]);

        int dup = 0;
        for (int j = 0; j < ws_count; j++) {
            if (strcmp(ws[j].word, tokens[i]) == 0) {
                if (score > ws[j].score) ws[j].score = score;
                dup = 1; break;
            }
        }
        if (!dup) {
            strncpy(ws[ws_count].word, tokens[i], MAX_WORD_LEN - 1);
            ws[ws_count].word[MAX_WORD_LEN - 1] = '\0';
            ws[ws_count].score = score;
            ws_count++;
        }
    }

    int take = (ws_count < TOP_N) ? ws_count : TOP_N;
    for (int i = 0; i < take; i++) {
        int mx = i;
        for (int j = i + 1; j < ws_count; j++)
            if (ws[j].score > ws[mx].score) mx = j;
        WordScore tmp = ws[i]; ws[i] = ws[mx]; ws[mx] = tmp;
    }

    float max_score = (take > 0) ? ws[0].score : 1.0f;

    printf("\n");
    print_separator('=', 60);
    printf("  TOP %d IMPORTANT WORDS  (ranked by attention output norm)\n", take);
    print_separator('=', 60);
    printf("  %-4s  %-20s  %-10s  %s\n", "Rank", "Word", "Score", "Relevance");
    print_separator('-', 60);
    for (int i = 0; i < take; i++) {
        int bar_len = (int)(20.0f * ws[i].score / max_score);
        char bar[22];
        for (int b = 0; b < 20; b++) bar[b] = (b < bar_len) ? '*' : ' ';
        bar[20] = '\0';
        printf("  #%-3d  %-20s  %10.4f  [%s]\n",
               i + 1, ws[i].word, ws[i].score, bar);
    }
    print_separator('=', 60);
    printf("\n");

    free(ws);
#undef TOP_N
}

/* ===========================================================================
 *                              MAIN
 * =========================================================================== */
int main() {
    char paragraph[MAX_PARAGRAPH_CHARS];
    char tokens[MAX_TOKENS][MAX_WORD_LEN];
    char sentences[MAX_SENTENCES][MAX_SENTENCE_LEN];
    int token_sentence[MAX_TOKENS];
    int n_sentences = 0;

    float (*embeddings)[EMBED_DIM]  = (float (*)[EMBED_DIM])malloc(MAX_TOKENS * sizeof(*embeddings));
    float (*attn_weights)[MAX_TOKENS] = (float (*)[MAX_TOKENS])malloc(MAX_TOKENS * sizeof(*attn_weights));
    float (*output)[EMBED_DIM]     = (float (*)[EMBED_DIM])malloc(MAX_TOKENS * sizeof(*output));
    SummaryMetrics metrics;
    char summary[MAX_SUMMARY_CHARS];

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        fprintf(stderr, "ERROR: No CUDA-capable GPU found.\n");
        return 1;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    printf("\n");
    print_separator('=', 70);
    printf("  Attention Summarizer: GloVe 300D + PCA Whitening  [CUDA]\n");
    printf("  GPU: %s  (SM %d.%d, %d SMs, %zu MB VRAM)\n",
           prop.name, prop.major, prop.minor,
           prop.multiProcessorCount,
           prop.totalGlobalMem / (1024 * 1024));
    printf("  Config: EMBED_DIM=%d  MAX_VOCAB=%d  NUM_HEADS=%d  HEAD_DIM=%d\n",
           EMBED_DIM, MAX_VOCAB, NUM_HEADS, HEAD_DIM);
    printf("  JACOBI_SWEEPS=%d  MAX_TOKENS=%d  BLOCK_SIZE=%d\n",
           JACOBI_SWEEPS, MAX_TOKENS, BLOCK_SIZE);
    print_separator('=', 70);

    double total_start = now_sec();

    if (!load_glove("glove.6B.300d.txt")) return 1;

    compute_pca_projections_cuda(total_start);

    printf("\nPaste paragraph (single line, end with Enter):\n");
    double io_wait_start = now_sec();
    if (fgets(paragraph, sizeof(paragraph), stdin) == NULL) {
        fprintf(stderr, "Error or EOF reading input\n");
        return 1;
    }
    double io_wait_time = now_sec() - io_wait_start;
    total_start += io_wait_time;

    int n_tokens = tokenize(paragraph, tokens, token_sentence, sentences, &n_sentences);
    printf("\n  [tokenize] %d tokens extracted from %d sentences\n", n_tokens, n_sentences);

    vectorize(tokens, n_tokens, embeddings);
    add_positional_encoding(embeddings, n_tokens);
    multihead_self_attention_cuda(embeddings, n_tokens, attn_weights, output);
    summarize(tokens, token_sentence, sentences, n_sentences,
              n_tokens, attn_weights, output, &metrics, summary);

    printf("\n%s\n", summary);

    print_top_words(tokens, n_tokens, output);

    print_comparison_report(tokens, token_sentence, n_tokens, n_sentences,
                            attn_weights, output, &metrics);

    print_report(total_start);

    free(vocab); free(embeddings); free(attn_weights); free(output);
    return 0;
}
