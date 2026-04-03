#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <ctype.h>

#define MAX_TOKENS     512
#define MAX_WORD_LEN   32
#define EMBED_DIM      200
#define MAX_VOCAB      500000
#define NUM_HEADS      8
#define HEAD_DIM       (EMBED_DIM / NUM_HEADS)
#define JACOBI_SWEEPS  100      
#define EIG_FLOOR      1e-6f  

typedef struct {
    char word[MAX_WORD_LEN];
    float vec[EMBED_DIM];
} Embedding;

Embedding *vocab;
int vocab_size = 0;

// PCA-derived projection matrices (computed once from GloVe corpus)
float W_Q[EMBED_DIM][EMBED_DIM];   // whitening: V * diag(1/sqrt(lambda))
float W_K[EMBED_DIM][EMBED_DIM];   // same as W_Q (symmetric semantic similarity)
// W_V is identity — no separate matrix needed

// ====================== JACOBI EIGENDECOMPOSITION ======================
// For a symmetric matrix A (200x200), finds eigenvectors V and eigenvalues d
// such that A = V * diag(d) * V^T
// Jacobi method: O(n^3) per sweep, ~50-100 sweeps for convergence
void jacobi_eigen(float A[EMBED_DIM][EMBED_DIM],
                  float V[EMBED_DIM][EMBED_DIM],
                  float d[EMBED_DIM]) {
    int n = EMBED_DIM;

    // Initialize V = Identity
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            V[i][j] = (i == j) ? 1.0f : 0.0f;

    // Copy A into working matrix (will be modified in-place toward diagonal)
    float S[EMBED_DIM][EMBED_DIM];
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            S[i][j] = A[i][j];

    for (int sweep = 0; sweep < JACOBI_SWEEPS; sweep++) {
        float off = 0.0f;
        for (int i = 0; i < n; i++)
            for (int j = i+1; j < n; j++)
                off += S[i][j] * S[i][j];
        if (off < 1e-10f) break;   // converged

        for (int p = 0; p < n-1; p++) {
            for (int q = p+1; q < n; q++) {
                if (fabsf(S[p][q]) < 1e-12f) continue;

                // Compute rotation angle
                float theta = 0.5f * atan2f(2.0f * S[p][q],
                                             S[p][p] - S[q][q]);
                float c = cosf(theta);
                float s = sinf(theta);

                // Apply Jacobi rotation to S (both rows/cols p and q)
                float new_Spp = c*c*S[p][p] + 2*s*c*S[p][q] + s*s*S[q][q];
                float new_Sqq = s*s*S[p][p] - 2*s*c*S[p][q] + c*c*S[q][q];
                float new_Spq = 0.0f;  // this is the off-diagonal we're zeroing

                for (int r = 0; r < n; r++) {
                    if (r == p || r == q) continue;
                    float new_rp = c*S[r][p] + s*S[r][q];
                    float new_rq = -s*S[r][p] + c*S[r][q];
                    S[r][p] = S[p][r] = new_rp;
                    S[r][q] = S[q][r] = new_rq;
                }
                S[p][p] = new_Spp;
                S[q][q] = new_Sqq;
                S[p][q] = S[q][p] = new_Spq;

                // Accumulate rotations into V
                for (int r = 0; r < n; r++) {
                    float new_rp =  c*V[r][p] + s*V[r][q];
                    float new_rq = -s*V[r][p] + c*V[r][q];
                    V[r][p] = new_rp;
                    V[r][q] = new_rq;
                }
            }
        }
    }

    // Extract eigenvalues from diagonal of S
    for (int i = 0; i < n; i++)
        d[i] = S[i][i];
}

// ====================== DERIVE W_Q, W_K FROM GLOVE PCA ======================
// Computes covariance of GloVe vectors, then whitening projection.
// W_Q = W_K = V * diag(1/sqrt(lambda))
// QK^T in this space = normalized semantic similarity (Mahalanobis-like)
void compute_pca_projections() {
    printf("Computing GloVe covariance matrix (%dx%d)...\n", EMBED_DIM, EMBED_DIM);

    // --- Step 1: Mean vector ---
    float mean[EMBED_DIM] = {0};
    for (int v = 0; v < vocab_size; v++)
        for (int d = 0; d < EMBED_DIM; d++)
            mean[d] += vocab[v].vec[d];
    for (int d = 0; d < EMBED_DIM; d++)
        mean[d] /= vocab_size;

    // --- Step 2: Covariance matrix (200x200) ---
    // Allocate on heap — 200*200*4 = 160KB, safe
    float (*cov)[EMBED_DIM] = calloc(EMBED_DIM, sizeof(*cov));
    if (!cov) { printf("ERROR: cov alloc failed\n"); return; }

    for (int v = 0; v < vocab_size; v++) {
        float centered[EMBED_DIM];
        for (int d = 0; d < EMBED_DIM; d++)
            centered[d] = vocab[v].vec[d] - mean[d];

        for (int i = 0; i < EMBED_DIM; i++)
            for (int j = i; j < EMBED_DIM; j++) {
                cov[i][j] += centered[i] * centered[j];
            }
    }
    // Normalize and symmetrize
    for (int i = 0; i < EMBED_DIM; i++)
        for (int j = i; j < EMBED_DIM; j++) {
            cov[i][j] /= (vocab_size - 1);
            cov[j][i]  = cov[i][j];
        }

    printf("Running Jacobi eigendecomposition (200x200)...\n");

    // --- Step 3: Eigendecompose ---
    float (*eigvecs)[EMBED_DIM] = malloc(EMBED_DIM * sizeof(*eigvecs));
    float eigvals[EMBED_DIM];
    if (!eigvecs) { printf("ERROR: eigvec alloc failed\n"); free(cov); return; }

    jacobi_eigen(cov, eigvecs, eigvals);

    // --- Step 4: Build whitening matrix W = V * diag(1/sqrt(lambda)) ---
    // W[i][j] = sum_k  eigvecs[i][k] * (1/sqrt(eigvals[k]))
    // But we want W such that projected x = x^T * W
    // So W[input_dim][output_dim] = eigvecs[output_dim][input_dim]^T * scale
    for (int i = 0; i < EMBED_DIM; i++) {
        for (int j = 0; j < EMBED_DIM; j++) {
            float scale = (eigvals[j] > EIG_FLOOR)
                          ? 1.0f / sqrtf(eigvals[j])
                          : 0.0f;
            // W_Q[i][j]: maps input dimension i → PCA dimension j
            W_Q[i][j] = eigvecs[i][j] * scale;
            W_K[i][j] = eigvecs[i][j] * scale;  // identical by design
        }
    }

    free(cov);
    free(eigvecs);
    printf("PCA projections ready. W_Q and W_K derived from real GloVe geometry.\n");
}

// ====================== PROJECT EMBEDDINGS ======================
void project_embeddings(float in[MAX_TOKENS][EMBED_DIM],
                        float W[EMBED_DIM][EMBED_DIM],
                        float out[MAX_TOKENS][EMBED_DIM],
                        int n_tokens) {
    for (int i = 0; i < n_tokens; i++) {
        for (int j = 0; j < EMBED_DIM; j++) {
            out[i][j] = 0.0f;
            for (int d = 0; d < EMBED_DIM; d++)
                out[i][j] += in[i][d] * W[d][j];
        }
    }
}

// ====================== LOAD GLOVE ======================
int load_glove(const char *filename) {
    FILE *f = fopen(filename, "r");
    if (!f) { printf("ERROR: Cannot open %s\n", filename); return 0; }

    vocab = (Embedding *)malloc(MAX_VOCAB * sizeof(Embedding));
    if (!vocab) return 0;

    char line[8192];
    vocab_size = 0;
    while (fgets(line, sizeof(line), f) && vocab_size < MAX_VOCAB) {
        char *token = strtok(line, " ");
        if (!token) continue;
        strncpy(vocab[vocab_size].word, token, MAX_WORD_LEN-1);
        vocab[vocab_size].word[MAX_WORD_LEN-1] = '\0';
        for (int d = 0; d < EMBED_DIM; d++) {
            token = strtok(NULL, " ");
            if (token) vocab[vocab_size].vec[d] = atof(token);
        }
        vocab_size++;
    }
    fclose(f);
    printf("Loaded %d GloVe vectors (%dD)\n", vocab_size, EMBED_DIM);
    return 1;
}

// ====================== TOKENIZE (with lowercase) ======================
int tokenize(char *text, char tokens[MAX_TOKENS][MAX_WORD_LEN]) {
    int n = 0;
    char *token = strtok(text, " .,;:!?\n\t\"'()[]{}");
    while (token && n < MAX_TOKENS) {
        strncpy(tokens[n], token, MAX_WORD_LEN-1);
        tokens[n][MAX_WORD_LEN-1] = '\0';
        // Lowercase — critical for GloVe match rate
        for (int k = 0; tokens[n][k]; k++)
            tokens[n][k] = tolower((unsigned char)tokens[n][k]);
        n++;
        token = strtok(NULL, " .,;:!?\n\t\"'()[]{}");
    }
    return n;
}

// ====================== VECTORIZE ======================
void vectorize(char tokens[MAX_TOKENS][MAX_WORD_LEN], int n_tokens,
               float embeddings[MAX_TOKENS][EMBED_DIM]) {
    int hit = 0;
    for (int i = 0; i < n_tokens; i++) {
        int found = 0;
        for (int v = 0; v < vocab_size; v++) {
            if (strcmp(tokens[i], vocab[v].word) == 0) {
                memcpy(embeddings[i], vocab[v].vec, sizeof(float)*EMBED_DIM);
                found = 1; hit++;
                break;
            }
        }
        if (!found) {
            memset(embeddings[i], 0, sizeof(float)*EMBED_DIM);
            printf("  [OOV] '%s' not in GloVe vocab\n", tokens[i]);
        }
    }
    printf("Vocab coverage: %d / %d tokens found\n", hit, n_tokens);
}

// ====================== POSITIONAL ENCODING ======================
void add_positional_encoding(float embeddings[MAX_TOKENS][EMBED_DIM], int n_tokens) {
    for (int pos = 0; pos < n_tokens; pos++) {
        for (int d = 0; d < EMBED_DIM; d++) {
            float angle = pos / powf(10000.0f, (2.0f*(d/2)) / (float)EMBED_DIM);
            embeddings[pos][d] += (d % 2 == 0) ? sinf(angle) : cosf(angle);
        }
    }
}

// ====================== MULTI-HEAD SELF-ATTENTION ======================
void multihead_self_attention(float embeddings[MAX_TOKENS][EMBED_DIM],
                              int n_tokens,
                              float attn_weights[MAX_TOKENS][MAX_TOKENS],
                              float output[MAX_TOKENS][EMBED_DIM]) {
    // Project using real PCA-derived matrices
    float (*Q)[EMBED_DIM] = malloc(MAX_TOKENS * sizeof(*Q));
    float (*K)[EMBED_DIM] = malloc(MAX_TOKENS * sizeof(*K));
    // V = identity: no projection, use embeddings directly

    project_embeddings(embeddings, W_Q, Q, n_tokens);
    project_embeddings(embeddings, W_K, K, n_tokens);

    float (*head_attn)[MAX_TOKENS][MAX_TOKENS] = malloc(NUM_HEADS * sizeof(*head_attn));
    if (!Q || !K || !head_attn) {
        printf("ERROR: attention alloc failed\n");
        free(Q); free(K); free(head_attn);
        return;
    }

    for (int h = 0; h < NUM_HEADS; h++) {
        int offset = h * HEAD_DIM;
        float scale = 1.0f / sqrtf((float)HEAD_DIM);

        // QK^T for this head
        for (int i = 0; i < n_tokens; i++) {
            for (int j = 0; j < n_tokens; j++) {
                float dot = 0.0f;
                for (int d = 0; d < HEAD_DIM; d++)
                    dot += Q[i][offset+d] * K[j][offset+d];
                head_attn[h][i][j] = dot * scale;
            }
        }

        // Stable softmax
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
        }

        // Attention output (V = embeddings directly, no W_V projection)
        for (int i = 0; i < n_tokens; i++)
            for (int d = 0; d < HEAD_DIM; d++) {
                float sum = 0.0f;
                for (int j = 0; j < n_tokens; j++)
                    sum += head_attn[h][i][j] * embeddings[j][offset+d];
                output[i][offset+d] = sum;
            }
    }

    // Average attention across heads
    memset(attn_weights, 0, sizeof(float)*MAX_TOKENS*MAX_TOKENS);
    for (int i = 0; i < n_tokens; i++)
        for (int j = 0; j < n_tokens; j++)
            for (int h = 0; h < NUM_HEADS; h++)
                attn_weights[i][j] += head_attn[h][i][j] / NUM_HEADS;

    free(Q); free(K); free(head_attn);
}

// ====================== TOKEN IMPORTANCE via OUTPUT NORM ======================
// After attention, tokens that received high focused attention from semantically
// related tokens will have output vectors with high L2 norm.
// This is a far more reliable importance signal than summing softmax rows.
float token_importance(float *output_vec) {
    float norm = 0.0f;
    for (int d = 0; d < EMBED_DIM; d++)
        norm += output_vec[d] * output_vec[d];
    return sqrtf(norm);
}

// ====================== SUMMARIZE (fixed sentence mapping) ======================
void summarize(char *original_text,
               char tokens[MAX_TOKENS][MAX_WORD_LEN],
               int n_tokens,
               float output[MAX_TOKENS][EMBED_DIM],
               char *summary) {

    // --- Split sentences first ---
    char sentences[30][1024];
    int n_sent = 0;
    char temp[8192];
    strncpy(temp, original_text, sizeof(temp)-1);
    char *sent = strtok(temp, ".!?");
    while (sent && n_sent < 30) {
        while (*sent == ' ' || *sent == '\n') sent++;
        if (strlen(sent) > 5) {
            strncpy(sentences[n_sent], sent, 1023);
            sentences[n_sent++][1023] = '\0';
        }
        sent = strtok(NULL, ".!?");
    }

    // --- Assign each token to a sentence (correct mapping) ---
    int token_sentence[MAX_TOKENS];
    memset(token_sentence, -1, sizeof(token_sentence));

    for (int i = 0; i < n_tokens; i++) {
        for (int s = 0; s < n_sent; s++) {
            // case-insensitive search
            char sent_lower[1024], tok_lower[MAX_WORD_LEN];
            strncpy(sent_lower, sentences[s], 1023); sent_lower[1023]='\0';
            strncpy(tok_lower,  tokens[i],    MAX_WORD_LEN-1); tok_lower[MAX_WORD_LEN-1]='\0';
            for (int k=0; sent_lower[k]; k++) sent_lower[k]=tolower((unsigned char)sent_lower[k]);
            for (int k=0; tok_lower[k];  k++) tok_lower[k] =tolower((unsigned char)tok_lower[k]);
            if (strstr(sent_lower, tok_lower)) {
                token_sentence[i] = s;
                break;
            }
        }
    }

    // --- Score sentences by average output-vector norm of their tokens ---
    float sent_score[30] = {0};
    int   sent_count[30] = {0};
    for (int i = 0; i < n_tokens; i++) {
        int s = token_sentence[i];
        if (s < 0) continue;
        sent_score[s] += token_importance(output[i]);
        sent_count[s]++;
    }
    for (int s = 0; s < n_sent; s++)
        if (sent_count[s] > 0)
            sent_score[s] /= sent_count[s];

    // --- Pick top 2 sentences ---
    int best[2] = {-1, -1};
    float bscore[2] = {-1.0f, -1.0f};
    for (int s = 0; s < n_sent; s++) {
        if (sent_score[s] > bscore[0])      { bscore[1]=bscore[0]; best[1]=best[0]; bscore[0]=sent_score[s]; best[0]=s; }
        else if (sent_score[s] > bscore[1]) { bscore[1]=sent_score[s]; best[1]=s; }
    }

    // Output in document order
    int order[2];
    if (best[0] != -1 && best[1] != -1)
        order[0] = (best[0] < best[1]) ? best[0] : best[1],
        order[1] = (best[0] < best[1]) ? best[1] : best[0];
    else order[0] = best[0], order[1] = best[1];

    summary[0] = '\0';
    strcat(summary, "=== SUMMARY (GloVe 200D + PCA Whitening Attention) ===\n");
    for (int k = 0; k < 2; k++)
        if (order[k] != -1)
            { strcat(summary, sentences[order[k]]); strcat(summary, ".\n"); }
    strcat(summary, "======================================================\n");
}

// ====================== MAIN ======================
int main() {
    char paragraph[8192], original[8192];
    char tokens[MAX_TOKENS][MAX_WORD_LEN];

    float (*embeddings)[EMBED_DIM] = malloc(MAX_TOKENS * sizeof(*embeddings));
    float (*attn_weights)[MAX_TOKENS] = malloc(MAX_TOKENS * sizeof(*attn_weights));
    float (*output)[EMBED_DIM] = malloc(MAX_TOKENS * sizeof(*output));
    char summary[4096];

    printf("=== Attention-Summarizer: GloVe 200D + PCA Whitening ===\n");

    if (!load_glove("glove.6B.200d.txt")) return 1;

    // Derive real projection matrices from GloVe corpus geometry
    compute_pca_projections();

    printf("\nPaste paragraph (single line, end with Enter):\n");
    fgets(paragraph, sizeof(paragraph), stdin);
    strncpy(original, paragraph, sizeof(original)-1);

    clock_t start = clock();

    int n_tokens = tokenize(paragraph, tokens);
    printf("Tokens: %d\n", n_tokens);

    vectorize(tokens, n_tokens, embeddings);
    add_positional_encoding(embeddings, n_tokens);
    multihead_self_attention(embeddings, n_tokens, attn_weights, output);
    summarize(original, tokens, n_tokens, output, summary);

    printf("\n%s\n", summary);
    printf("Time: %.3fs\n", (double)(clock()-start)/CLOCKS_PER_SEC);

    free(vocab); free(embeddings); free(attn_weights); free(output);
    return 0;
}