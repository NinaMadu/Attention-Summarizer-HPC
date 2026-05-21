#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <ctype.h>
#include <mpi.h>

#define MAX_TOKENS     512
#define MAX_WORD_LEN   32
#define MAX_SENTENCES  30
#define EMBED_DIM      200
#define MAX_VOCAB      500000
#define NUM_HEADS      8
#define HEAD_DIM       (EMBED_DIM / NUM_HEADS)
#define JACOBI_SWEEPS  100
#define EIG_FLOOR      1e-6f

/* ====================== LOGGING UTILITIES ====================== */

typedef struct {
    const char *name;
    double elapsed_sec;
    long long ops;
} StageLog;

#define MAX_STAGES 16
static StageLog stage_log[MAX_STAGES];
static int stage_count = 0;

static double now_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

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
    printf("  COMPLEXITY & TIMING REPORT\n");
    print_separator('=', 70);
    printf("  %-32s  %10s  %18s  %s\n", "Stage", "Time (s)", "Operations", "% Total");
    print_separator('-', 70);
    for (int i = 0; i < stage_count; i++) {
        double pct = (total > 0) ? 100.0 * stage_log[i].elapsed_sec / total : 0.0;
        /* bar: 20 chars wide */
        int bar_len = (int)(pct / 5.0);   /* 1 char = 5% */
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
    printf("    covariance_matrix    O(V x D^2)        ~%.1fB multiply-adds\n",
           (double)MAX_VOCAB * EMBED_DIM * EMBED_DIM / 2.0 / 1e9);
    printf("    jacobi_eigen         O(sweeps x D^3)   sweeps=%d => ~%.0fM rotations\n",
           JACOBI_SWEEPS,
           (double)JACOBI_SWEEPS * EMBED_DIM * EMBED_DIM / 2.0 / 1e6);
    printf("    project_embeddings   O(T x D^2)        T=n_tokens\n");
    printf("    self_attention QK^T  O(H x T^2 x d)    H=%d heads, d=%d\n", NUM_HEADS, HEAD_DIM);
    printf("    self_attention out   O(H x T^2 x d)\n");
    print_separator('=', 70);
    printf("\n");
}

/* ====================== JACOBI PROGRESS LOGGING ====================== */
/* Reports convergence every 10 sweeps so you can see the off-diagonal
   norm decreasing — key to justifying JACOBI_SWEEPS in HPC context. */
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

Embedding *vocab;
int vocab_size = 0;

float W_Q[EMBED_DIM][EMBED_DIM];
float W_K[EMBED_DIM][EMBED_DIM];

int mpi_rank = 0;
int mpi_size = 1;

static int is_root() {
    return mpi_rank == 0;
}

static void mpi_range(int n, int *start, int *end) {
    int base = n / mpi_size;
    int rem = n % mpi_size;
    *start = mpi_rank * base + ((mpi_rank < rem) ? mpi_rank : rem);
    *end = *start + base + ((mpi_rank < rem) ? 1 : 0);
}

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

/* ====================== PCA PROJECTIONS ====================== */
void compute_pca_projections(double total_start) {
    (void)total_start;
    if (is_root()) {
        printf("\n");
        print_separator('-', 60);
        printf("  STAGE: covariance_matrix (MPI)\n");
        printf("  Complexity: O(V x D^2)  =>  V=%d, D=%d, ranks=%d\n",
               vocab_size, EMBED_DIM, mpi_size);
        printf("  Expected ops: ~%.2fB\n",
               (double)vocab_size * EMBED_DIM * EMBED_DIM / 2.0 / 1e9);
        print_separator('-', 60);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    int start, end;
    mpi_range(vocab_size, &start, &end);

    float local_mean[EMBED_DIM] = {0};
    float mean[EMBED_DIM] = {0};
    for (int v = start; v < end; v++)
        for (int dd = 0; dd < EMBED_DIM; dd++)
            local_mean[dd] += vocab[v].vec[dd];

    MPI_Reduce(local_mean, mean, EMBED_DIM, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (is_root())
        for (int dd = 0; dd < EMBED_DIM; dd++)
            mean[dd] /= vocab_size;
    MPI_Bcast(mean, EMBED_DIM, MPI_FLOAT, 0, MPI_COMM_WORLD);

    float (*local_cov)[EMBED_DIM] = calloc(EMBED_DIM, sizeof(*local_cov));
    float (*cov)[EMBED_DIM] = NULL;
    if (is_root()) cov = calloc(EMBED_DIM, sizeof(*cov));
    if (!local_cov || (is_root() && !cov)) {
        if (is_root()) printf("ERROR: cov alloc failed\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    long long local_cov_ops = 0;
    for (int v = start; v < end; v++) {
        float centered[EMBED_DIM];
        for (int dd = 0; dd < EMBED_DIM; dd++)
            centered[dd] = vocab[v].vec[dd] - mean[dd];
        for (int i = 0; i < EMBED_DIM; i++)
            for (int j = i; j < EMBED_DIM; j++) {
                local_cov[i][j] += centered[i] * centered[j];
                local_cov_ops++;
            }
    }

    MPI_Reduce(local_cov, cov, EMBED_DIM * EMBED_DIM, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    long long cov_ops = 0;
    MPI_Reduce(&local_cov_ops, &cov_ops, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    free(local_cov);

    if (is_root()) {
        for (int i = 0; i < EMBED_DIM; i++)
            for (int j = i; j < EMBED_DIM; j++) {
                cov[i][j] /= (vocab_size - 1);
                cov[j][i]  = cov[i][j];
            }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();
    if (is_root()) {
        log_stage("covariance_matrix_mpi", t0, t1, cov_ops);
        printf("  [cov] MPI finished in %.4fs  total ops: %lld\n", t1-t0, cov_ops);
    }

    if (is_root()) {
        printf("\n");
        print_separator('-', 60);
        printf("  STAGE: jacobi_eigen\n");
        printf("  Complexity: O(sweeps x D^3)  =>  ~%.0fM Jacobi rotations\n",
               (double)JACOBI_SWEEPS * EMBED_DIM * EMBED_DIM / 2.0 / 1e6);
        print_separator('-', 60);

        double t2 = MPI_Wtime();
        float (*eigvecs)[EMBED_DIM] = malloc(EMBED_DIM * sizeof(*eigvecs));
        float eigvals[EMBED_DIM];
        if (!eigvecs) {
            printf("ERROR: eigvec alloc failed\n");
            free(cov);
            MPI_Abort(MPI_COMM_WORLD, 3);
        }

        long long jacobi_ops = 0;
        jacobi_eigen(cov, eigvecs, eigvals, &jacobi_ops);

        double t3 = MPI_Wtime();
        log_stage("jacobi_eigen", t2, t3, jacobi_ops);
        printf("  [jacobi] Finished in %.4fs\n", t3-t2);

        printf("\n");
        print_separator('-', 60);
        printf("  STAGE: build_whitening_matrices\n");
        printf("  Complexity: O(D^2) = %d ops\n", EMBED_DIM * EMBED_DIM);
        print_separator('-', 60);

        double t4 = MPI_Wtime();
        long long w_ops = 0;
        printf("  [pca] Top 5 eigenvalues (variance explained):\n");
        float ev_copy[EMBED_DIM];
        memcpy(ev_copy, eigvals, sizeof(eigvals));
        for (int pass = 0; pass < 5; pass++) {
            int mx = pass;
            for (int k = pass+1; k < EMBED_DIM; k++)
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
        double t5 = MPI_Wtime();
        log_stage("build_whitening_W", t4, t5, w_ops);
        printf("  [pca] W_Q / W_K built in %.6fs\n", t5-t4);

        free(cov);
        free(eigvecs);
    }

    MPI_Bcast(W_Q, EMBED_DIM * EMBED_DIM, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(W_K, EMBED_DIM * EMBED_DIM, MPI_FLOAT, 0, MPI_COMM_WORLD);
}

/* ====================== PROJECT EMBEDDINGS ====================== */
void project_embeddings(float in[MAX_TOKENS][EMBED_DIM],
                        float W[EMBED_DIM][EMBED_DIM],
                        float out[MAX_TOKENS][EMBED_DIM],
                        int n_tokens,
                        long long *ops) {
    *ops = 0;
    for (int i = 0; i < n_tokens; i++) {
        for (int j = 0; j < EMBED_DIM; j++) {
            out[i][j] = 0.0f;
            for (int dd = 0; dd < EMBED_DIM; dd++) {
                out[i][j] += in[i][dd] * W[dd][j];
                (*ops)++;
            }
        }
    }
}

/* ====================== LOAD GLOVE ====================== */
int load_glove(const char *filename) {
    if (is_root()) {
        printf("\n");
        print_separator('-', 60);
        printf("  STAGE: load_glove (MPI replicated)\n");
        printf("  Complexity: O(V x D) reads per rank, V<=%d, D=%d, ranks=%d\n",
               MAX_VOCAB, EMBED_DIM, mpi_size);
        printf("  Accuracy mode: every rank reads the same file in serial order\n");
        print_separator('-', 60);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    FILE *f = fopen(filename, "r");
    if (!f) {
        printf("ERROR: rank %d cannot open %s\n", mpi_rank, filename);
        return 0;
    }

    vocab = (Embedding *)malloc(MAX_VOCAB * sizeof(Embedding));
    if (!vocab) {
        printf("ERROR: rank %d vocab allocation failed\n", mpi_rank);
        fclose(f);
        return 0;
    }

    char line[8192];
    vocab_size = 0;
    while (fgets(line, sizeof(line), f) && vocab_size < MAX_VOCAB) {
        char *token = strtok(line, " ");
        if (!token) continue;
        strncpy(vocab[vocab_size].word, token, MAX_WORD_LEN-1);
        vocab[vocab_size].word[MAX_WORD_LEN-1] = '\0';
        for (int dd = 0; dd < EMBED_DIM; dd++) {
            token = strtok(NULL, " ");
            if (token) vocab[vocab_size].vec[dd] = atof(token);
        }
        vocab_size++;
        if (is_root() && vocab_size % 100000 == 0)
            printf("  [load] Loaded %d vectors so far...\n", vocab_size);
    }
    fclose(f);

    double t1 = MPI_Wtime();
    long long load_ops = (long long)vocab_size * EMBED_DIM;
    if (is_root()) {
        log_stage("load_glove_replicated", t0, t1, load_ops);
        printf("  [load] Loaded %d GloVe vectors (%dD) per rank in %.4fs\n",
               vocab_size, EMBED_DIM, t1-t0);
    }
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
             char sentences[MAX_SENTENCES][1024],
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
            if (sent_len < 1023) {
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
    if (is_root()) {
        printf("\n");
        print_separator('-', 60);
        printf("  STAGE: vectorize (MPI)\n");
        printf("  Complexity: O(T x V) linear scan,  T=%d, V=%d, ranks=%d\n",
               n_tokens, vocab_size, mpi_size);
        printf("  Worst-case comparisons: %lld\n", (long long)n_tokens * vocab_size);
        print_separator('-', 60);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    int start, end;
    mpi_range(n_tokens, &start, &end);
    float (*local_embeddings)[EMBED_DIM] = calloc(MAX_TOKENS, sizeof(*local_embeddings));
    int local_found[MAX_TOKENS] = {0};
    int found_flags[MAX_TOKENS] = {0};
    if (!local_embeddings) {
        if (is_root()) printf("ERROR: vectorize local allocation failed\n");
        MPI_Abort(MPI_COMM_WORLD, 5);
    }

    int local_hit = 0;
    long long local_cmp_ops = 0;
    for (int i = start; i < end; i++) {
        int found = 0;
        for (int v = 0; v < vocab_size; v++) {
            local_cmp_ops++;
            if (strcmp(tokens[i], vocab[v].word) == 0) {
                memcpy(local_embeddings[i], vocab[v].vec, sizeof(float)*EMBED_DIM);
                found = 1;
                local_found[i] = 1;
                local_hit++;
                break;
            }
        }
        if (!found)
            memset(local_embeddings[i], 0, sizeof(float)*EMBED_DIM);
    }

    MPI_Reduce(local_embeddings, embeddings, MAX_TOKENS * EMBED_DIM,
               MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(local_found, found_flags, MAX_TOKENS,
               MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    int hit = 0;
    long long cmp_ops = 0;
    MPI_Reduce(&local_hit, &hit, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_cmp_ops, &cmp_ops, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    free(local_embeddings);

    MPI_Bcast(embeddings, MAX_TOKENS * EMBED_DIM, MPI_FLOAT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();
    if (is_root()) {
        for (int i = 0; i < n_tokens; i++)
            if (!found_flags[i])
                printf("  [oov] '%s' not in GloVe vocab\n", tokens[i]);
        log_stage("tokenize+vectorize_mpi", t0, t1, cmp_ops);
        printf("  [vec] Coverage: %d/%d tokens hit  comparisons: %lld  time: %.4fs\n",
               hit, n_tokens, cmp_ops, t1-t0);
    }
}

/* ====================== POSITIONAL ENCODING ====================== */
void add_positional_encoding(float embeddings[MAX_TOKENS][EMBED_DIM], int n_tokens) {
    if (is_root()) {
        printf("\n");
        print_separator('-', 60);
        printf("  STAGE: positional_encoding\n");
        printf("  Complexity: O(T x D)  ops: %d\n", n_tokens * EMBED_DIM);
        print_separator('-', 60);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    long long ops = 0;
    for (int pos = 0; pos < n_tokens; pos++) {
        for (int dd = 0; dd < EMBED_DIM; dd++) {
            float angle = pos / powf(10000.0f, (2.0f*(dd/2)) / (float)EMBED_DIM);
            embeddings[pos][dd] += (dd % 2 == 0) ? sinf(angle) : cosf(angle);
            ops++;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();
    if (is_root()) {
        log_stage("positional_encoding", t0, t1, ops);
        printf("  [pe] Done in %.6fs  ops: %lld\n", t1-t0, ops);
    }
}

/* ====================== MULTI-HEAD SELF-ATTENTION ====================== */
void multihead_self_attention(float embeddings[MAX_TOKENS][EMBED_DIM],
                              int n_tokens,
                              float attn_weights[MAX_TOKENS][MAX_TOKENS],
                              float output[MAX_TOKENS][EMBED_DIM]) {
    printf("\n");
    print_separator('-', 60);
    printf("  STAGE: multi_head_self_attention\n");
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

    float (*Q)[EMBED_DIM] = malloc(MAX_TOKENS * sizeof(*Q));
    float (*K)[EMBED_DIM] = malloc(MAX_TOKENS * sizeof(*K));
    float (*head_attn)[MAX_TOKENS][MAX_TOKENS] = malloc(NUM_HEADS * sizeof(*head_attn));

    if (!Q || !K || !head_attn) {
        printf("ERROR: attention alloc failed\n");
        free(Q); free(K); free(head_attn);
        return;
    }

    long long proj_ops = 0, qk_ops = 0, out_ops = 0;

    printf("  [attn] Projecting Q...\n");
    double tp0 = now_sec();
    project_embeddings(embeddings, W_Q, Q, n_tokens, &proj_ops);
    double tp1 = now_sec();
    printf("  [attn] Q projected in %.4fs  (%lld multiply-adds)\n", tp1-tp0, proj_ops);

    long long k_ops = 0;
    printf("  [attn] Projecting K...\n");
    project_embeddings(embeddings, W_K, K, n_tokens, &k_ops);
    double tp2 = now_sec();
    printf("  [attn] K projected in %.4fs  (%lld multiply-adds)\n", tp2-tp1, k_ops);
    proj_ops += k_ops;

    float scale = 1.0f / sqrtf((float)HEAD_DIM);
    printf("  [attn] Attention scale factor: 1/sqrt(%d) = %.6f\n", HEAD_DIM, scale);

    for (int h = 0; h < NUM_HEADS; h++) {
        int offset = h * HEAD_DIM;

        for (int i = 0; i < n_tokens; i++) {
            for (int j = 0; j < n_tokens; j++) {
                float dot = 0.0f;
                for (int dd = 0; dd < HEAD_DIM; dd++)
                    dot += Q[i][offset+dd] * K[j][offset+dd];
                head_attn[h][i][j] = dot * scale;
                qk_ops += HEAD_DIM;
            }
        }

        /* Log head-level stats: mean attention entropy */
        float mean_max = 0.0f;
        for (int i = 0; i < n_tokens; i++) {
            float mx = head_attn[h][i][0];
            for (int j = 1; j < n_tokens; j++)
                if (head_attn[h][i][j] > mx) mx = head_attn[h][i][j];
            float sum = 0.0f;
            for (int j = 0; j < n_tokens; j++) {
                head_attn[h][i][j] = expf(head_attn[h][i][j] - mx);
                sum += head_attn[h][i][j];
            }
            for (int j = 0; j < n_tokens; j++)
                head_attn[h][i][j] /= sum;
            mean_max += head_attn[h][i][i];  /* diagonal = self-attention */
        }
        mean_max /= n_tokens;
        printf("  [attn] head %d/%d  avg self-attention weight = %.4f\n",
               h+1, NUM_HEADS, mean_max);

        for (int i = 0; i < n_tokens; i++)
            for (int dd = 0; dd < HEAD_DIM; dd++) {
                float sum = 0.0f;
                for (int j = 0; j < n_tokens; j++)
                    sum += head_attn[h][i][j] * embeddings[j][offset+dd];
                output[i][offset+dd] = sum;
                out_ops += n_tokens;
            }
    }

    memset(attn_weights, 0, sizeof(float)*MAX_TOKENS*MAX_TOKENS);
    for (int i = 0; i < n_tokens; i++)
        for (int j = 0; j < n_tokens; j++)
            for (int h = 0; h < NUM_HEADS; h++)
                attn_weights[i][j] += head_attn[h][i][j] / NUM_HEADS;

    double t1 = now_sec();
    long long total_attn_ops = proj_ops + qk_ops + out_ops;
    log_stage("self_attention", t0, t1, total_attn_ops);
    printf("  [attn] Total ops — project:%lld  QKt:%lld  output:%lld  sum:%lld\n",
           proj_ops, qk_ops, out_ops, total_attn_ops);
    printf("  [attn] Attention stage finished in %.4fs\n", t1-t0);

    free(Q); free(K); free(head_attn);
}

void multihead_self_attention_mpi(float embeddings[MAX_TOKENS][EMBED_DIM],
                                  int n_tokens,
                                  float attn_weights[MAX_TOKENS][MAX_TOKENS],
                                  float output[MAX_TOKENS][EMBED_DIM]) {
    if (is_root()) {
        printf("\n");
        print_separator('-', 60);
        printf("  STAGE: multi_head_self_attention (MPI)\n");
        printf("  T=%d  H=%d heads  head_dim d=%d  ranks=%d\n",
               n_tokens, NUM_HEADS, HEAD_DIM, mpi_size);
        printf("  Project Q,K:  O(T x D^2) = %lld ops each\n",
               (long long)n_tokens * EMBED_DIM * EMBED_DIM);
        printf("  QK^T:         O(H x T^2 x d) = %lld ops\n",
               (long long)NUM_HEADS * n_tokens * n_tokens * HEAD_DIM);
        printf("  Softmax:      O(H x T^2) = %d ops\n", NUM_HEADS * n_tokens * n_tokens);
        printf("  Output V:     O(H x T^2 x d) = %lld ops\n",
               (long long)NUM_HEADS * n_tokens * n_tokens * HEAD_DIM);
        print_separator('-', 60);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    int start, end;
    mpi_range(n_tokens, &start, &end);

    float (*Q)[EMBED_DIM] = calloc(MAX_TOKENS, sizeof(*Q));
    float (*K)[EMBED_DIM] = calloc(MAX_TOKENS, sizeof(*K));
    float (*local_attn)[MAX_TOKENS] = calloc(MAX_TOKENS, sizeof(*local_attn));
    float (*local_output)[EMBED_DIM] = calloc(MAX_TOKENS, sizeof(*local_output));

    if (!Q || !K || !local_attn || !local_output) {
        if (is_root()) printf("ERROR: attention alloc failed\n");
        free(Q); free(K); free(local_attn); free(local_output);
        MPI_Abort(MPI_COMM_WORLD, 6);
    }

    long long local_proj_ops = 0, local_qk_ops = 0, local_out_ops = 0;

    double tp0 = MPI_Wtime();
    for (int i = start; i < end; i++) {
        for (int j = 0; j < EMBED_DIM; j++) {
            for (int dd = 0; dd < EMBED_DIM; dd++) {
                Q[i][j] += embeddings[i][dd] * W_Q[dd][j];
                K[i][j] += embeddings[i][dd] * W_K[dd][j];
                local_proj_ops += 2;
            }
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, Q, MAX_TOKENS * EMBED_DIM, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, K, MAX_TOKENS * EMBED_DIM, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    double tp1 = MPI_Wtime();

    float scale = 1.0f / sqrtf((float)HEAD_DIM);
    if (is_root()) {
        printf("  [attn] Q/K projected in %.4fs\n", tp1-tp0);
        printf("  [attn] Attention scale factor: 1/sqrt(%d) = %.6f\n", HEAD_DIM, scale);
    }

    float self_local[NUM_HEADS] = {0};
    float self_global[NUM_HEADS] = {0};
    float scores[MAX_TOKENS];

    for (int h = 0; h < NUM_HEADS; h++) {
        int offset = h * HEAD_DIM;

        for (int i = start; i < end; i++) {
            for (int j = 0; j < n_tokens; j++) {
                float dot = 0.0f;
                for (int dd = 0; dd < HEAD_DIM; dd++)
                    dot += Q[i][offset+dd] * K[j][offset+dd];
                scores[j] = dot * scale;
                local_qk_ops += HEAD_DIM;
            }

            float mx = scores[0];
            for (int j = 1; j < n_tokens; j++)
                if (scores[j] > mx) mx = scores[j];

            float softmax_sum = 0.0f;
            for (int j = 0; j < n_tokens; j++) {
                scores[j] = expf(scores[j] - mx);
                softmax_sum += scores[j];
            }
            for (int j = 0; j < n_tokens; j++)
                scores[j] /= softmax_sum;

            self_local[h] += scores[i];

            for (int j = 0; j < n_tokens; j++)
                local_attn[i][j] += scores[j] / NUM_HEADS;

            for (int dd = 0; dd < HEAD_DIM; dd++) {
                float sum = 0.0f;
                for (int j = 0; j < n_tokens; j++)
                    sum += scores[j] * embeddings[j][offset+dd];
                local_output[i][offset+dd] = sum;
                local_out_ops += n_tokens;
            }
        }
    }

    MPI_Reduce(local_attn, attn_weights, MAX_TOKENS * MAX_TOKENS,
               MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(local_output, output, MAX_TOKENS * EMBED_DIM,
               MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(self_local, self_global, NUM_HEADS,
               MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    long long proj_ops = 0, qk_ops = 0, out_ops = 0;
    MPI_Reduce(&local_proj_ops, &proj_ops, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_qk_ops, &qk_ops, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_out_ops, &out_ops, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();
    if (is_root()) {
        for (int h = 0; h < NUM_HEADS; h++) {
            float mean_self = (n_tokens > 0) ? self_global[h] / n_tokens : 0.0f;
            printf("  [attn] head %d/%d  avg self-attention weight = %.4f\n",
                   h+1, NUM_HEADS, mean_self);
        }
        long long total_attn_ops = proj_ops + qk_ops + out_ops;
        log_stage("self_attention_mpi", t0, t1, total_attn_ops);
        printf("  [attn] Total ops - project:%lld  QKt:%lld  output:%lld  sum:%lld\n",
               proj_ops, qk_ops, out_ops, total_attn_ops);
        printf("  [attn] Attention stage finished in %.4fs\n", t1-t0);
    }

    free(Q);
    free(K);
    free(local_attn);
    free(local_output);
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
               char sentences[MAX_SENTENCES][1024],
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
    printf("  [sum] Finished in %.6fs\n", t1-t0);

    summary[0] = '\0';
    strcat(summary, "=== SUMMARY (GloVe 200D + PCA Whitening Attention Centrality) ===\n");
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
           attn_checksum, attn_abs_checksum, attn_min, attn_max, row_sum_min, row_sum_max);
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
    for (int i = 0; i < sample; i++) {
        for (int j = 0; j < sample; j++) {
            printf("%s%.9f", (j == 0) ? "" : ",", attn_weights[i][j]);
        }
        printf("\n");
    }
    print_separator('=', 70);
}

/* ====================== TOP-10 IMPORTANT WORDS ====================== */
/* Ranks tokens by their output-vector L2 norm as an auxiliary diagnostic.
   Stop-words are filtered so the result shows
   semantically meaningful words, not "the", "a", "is", etc.
   Deduplicates so the same word only appears once in the ranking.    */

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
    /* Build scored list, skip stop-words, deduplicate */
    typedef struct { char word[MAX_WORD_LEN]; float score; } WordScore;
    WordScore *ws = malloc(n_tokens * sizeof(WordScore));
    if (!ws) return;
    int ws_count = 0;

    for (int i = 0; i < n_tokens; i++) {
        if (is_stopword(tokens[i])) continue;
        if (strlen(tokens[i]) < 2)  continue;

        float score = token_importance(output[i]);

        /* Deduplicate: if already present, keep the higher score */
        int dup = 0;
        for (int j = 0; j < ws_count; j++) {
            if (strcmp(ws[j].word, tokens[i]) == 0) {
                if (score > ws[j].score) ws[j].score = score;
                dup = 1; break;
            }
        }
        if (!dup) {
            strncpy(ws[ws_count].word, tokens[i], MAX_WORD_LEN-1);
            ws[ws_count].word[MAX_WORD_LEN-1] = '\0';
            ws[ws_count].score = score;
            ws_count++;
        }
    }

    /* Partial selection sort for top TOP_N */
    int take = (ws_count < TOP_N) ? ws_count : TOP_N;
    for (int i = 0; i < take; i++) {
        int mx = i;
        for (int j = i+1; j < ws_count; j++)
            if (ws[j].score > ws[mx].score) mx = j;
        WordScore tmp = ws[i]; ws[i] = ws[mx]; ws[mx] = tmp;
    }

    /* Find max score for bar scaling */
    float max_score = (take > 0) ? ws[0].score : 1.0f;

    printf("\n");
    print_separator('=', 60);
    printf("  TOP %d IMPORTANT WORDS  (ranked by attention output norm)\n", take);
    print_separator('=', 60);
    printf("  %-4s  %-20s  %-10s  %s\n", "Rank", "Word", "Score", "Relevance");
    print_separator('-', 60);
    for (int i = 0; i < take; i++) {
        /* ASCII bar: 20 chars wide */
        int bar_len = (int)(20.0f * ws[i].score / max_score);
        char bar[22];
        for (int b = 0; b < 20; b++) bar[b] = (b < bar_len) ? '*' : ' ';
        bar[20] = '\0';
        printf("  #%-3d  %-20s  %10.4f  [%s]\n",
               i+1, ws[i].word, ws[i].score, bar);
    }
    print_separator('=', 60);
    printf("\n");

    free(ws);
#undef TOP_N
}

/* ====================== MAIN ====================== */
int main() {
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    char paragraph[8192];
    char tokens[MAX_TOKENS][MAX_WORD_LEN];
    char sentences[MAX_SENTENCES][1024];
    int token_sentence[MAX_TOKENS];
    int n_tokens = 0;
    int n_sentences = 0;

    float (*embeddings)[EMBED_DIM] = malloc(MAX_TOKENS * sizeof(*embeddings));
    float (*attn_weights)[MAX_TOKENS] = malloc(MAX_TOKENS * sizeof(*attn_weights));
    float (*output)[EMBED_DIM] = malloc(MAX_TOKENS * sizeof(*output));
    SummaryMetrics metrics;
    char summary[4096];

    if (!embeddings || !attn_weights || !output) {
        if (is_root()) printf("ERROR: main allocation failed\n");
        MPI_Abort(MPI_COMM_WORLD, 7);
    }

    if (is_root()) {
        printf("\n");
        print_separator('=', 70);
        printf("  MPI Attention Summarizer: GloVe 200D + PCA Whitening\n");
        printf("  Config: EMBED_DIM=%d  MAX_VOCAB=%d  NUM_HEADS=%d  HEAD_DIM=%d  MPI_RANKS=%d\n",
               EMBED_DIM, MAX_VOCAB, NUM_HEADS, HEAD_DIM, mpi_size);
        printf("  JACOBI_SWEEPS=%d  MAX_TOKENS=%d\n", JACOBI_SWEEPS, MAX_TOKENS);
        print_separator('=', 70);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double total_start = now_sec();

    int load_ok = load_glove("glove.6B.200d.txt");
    int all_load_ok = 0;
    MPI_Allreduce(&load_ok, &all_load_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (!all_load_ok) {
        MPI_Finalize();
        return 1;
    }

    int min_vocab_size = 0, max_vocab_size = 0;
    MPI_Allreduce(&vocab_size, &min_vocab_size, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&vocab_size, &max_vocab_size, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (min_vocab_size != max_vocab_size) {
        if (is_root())
            printf("ERROR: MPI ranks loaded different vocab sizes: min=%d max=%d\n",
                   min_vocab_size, max_vocab_size);
        MPI_Finalize();
        return 1;
    }

    compute_pca_projections(total_start);

    int input_ok = 1;
    if (is_root()) {
        printf("\nPaste paragraph (single line, end with Enter):\n");
        if (!fgets(paragraph, sizeof(paragraph), stdin)) {
            printf("ERROR: failed to read input paragraph\n");
            input_ok = 0;
        }

        if (input_ok) {
            n_tokens = tokenize(paragraph, tokens, token_sentence, sentences, &n_sentences);
            printf("\n  [tokenize] %d tokens extracted from %d sentences\n", n_tokens, n_sentences);
        }
    }
    MPI_Bcast(&input_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!input_ok) {
        free(vocab); free(embeddings); free(attn_weights); free(output);
        MPI_Finalize();
        return 1;
    }

    MPI_Bcast(&n_tokens, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n_sentences, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(tokens, MAX_TOKENS * MAX_WORD_LEN, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(token_sentence, MAX_TOKENS, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(sentences, MAX_SENTENCES * 1024, MPI_CHAR, 0, MPI_COMM_WORLD);

    vectorize(tokens, n_tokens, embeddings);
    add_positional_encoding(embeddings, n_tokens);
    multihead_self_attention_mpi(embeddings, n_tokens, attn_weights, output);

    if (is_root()) {
        summarize(tokens, token_sentence, sentences, n_sentences,
                  n_tokens, attn_weights, output, &metrics, summary);

        printf("\n%s\n", summary);

        /* Top-10 most important words by attention output norm */
        print_top_words(tokens, n_tokens, output);

        /* Deterministic numeric output for comparing serial/MPI/OpenMP/CUDA runs */
        print_comparison_report(tokens, token_sentence, n_tokens, n_sentences,
                                attn_weights, output, &metrics);

        /* Print the full timing + complexity report */
        print_report(total_start);
    }

    free(vocab); free(embeddings); free(attn_weights); free(output);
    MPI_Finalize();
    return 0;
}
