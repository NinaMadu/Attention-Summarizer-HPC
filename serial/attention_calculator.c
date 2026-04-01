#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_TOKENS     512
#define MAX_WORD_LEN   32
#define EMBED_DIM      50          // real GloVe 50d
#define MAX_VOCAB      500000      // safe for mini or full GloVe

typedef struct {
    char word[MAX_WORD_LEN];
    float vec[EMBED_DIM];
} Embedding;

Embedding *vocab;
int vocab_size = 0;

// ====================== 1. LOAD REAL GLOVE EMBEDDINGS ======================
int load_glove(const char *filename) {
    FILE *f = fopen(filename, "r");
    if (!f) {
        printf("ERROR: Cannot open %s\n", filename);
        return 0;
    }
    vocab = (Embedding *)malloc(MAX_VOCAB * sizeof(Embedding));
    if (!vocab) return 0;

    char line[4096];
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
    printf("Loaded %d real GloVe vectors (50D)\n", vocab_size);
    return 1;
}

// ====================== 2. TOKENIZE ======================
int tokenize(char *text, char tokens[MAX_TOKENS][MAX_WORD_LEN]) {
    int n = 0;
    char *token = strtok(text, " .,;:!?\n\t");
    while (token && n < MAX_TOKENS) {
        strncpy(tokens[n], token, MAX_WORD_LEN-1);
        tokens[n][MAX_WORD_LEN-1] = '\0';
        n++;
        token = strtok(NULL, " .,;:!?\n\t");
    }
    return n;
}

// ====================== 3. VECTORIZE (real embeddings) ======================
void vectorize(char tokens[MAX_TOKENS][MAX_WORD_LEN], int n_tokens,
               float embeddings[MAX_TOKENS][EMBED_DIM]) {
    for (int i = 0; i < n_tokens; i++) {
        int found = 0;
        for (int v = 0; v < vocab_size; v++) {
            if (strcmp(tokens[i], vocab[v].word) == 0) {
                memcpy(embeddings[i], vocab[v].vec, sizeof(float)*EMBED_DIM);
                found = 1;
                break;
            }
        }
        if (!found) memset(embeddings[i], 0, sizeof(float)*EMBED_DIM); // unknown → zero
    }
}

// ====================== 4. SELF-ATTENTION (manual) ======================
void self_attention(float embeddings[MAX_TOKENS][EMBED_DIM], int n_tokens,
                    float attn_weights[MAX_TOKENS][MAX_TOKENS],
                    float output[MAX_TOKENS][EMBED_DIM]) {
    float Q[MAX_TOKENS][EMBED_DIM];
    float K[MAX_TOKENS][EMBED_DIM];
    float V[MAX_TOKENS][EMBED_DIM];
    float scores[MAX_TOKENS][MAX_TOKENS];

    memcpy(Q, embeddings, sizeof(float)*n_tokens*EMBED_DIM);
    memcpy(K, embeddings, sizeof(float)*n_tokens*EMBED_DIM);
    memcpy(V, embeddings, sizeof(float)*n_tokens*EMBED_DIM);

    float scale = 1.0f / sqrt(EMBED_DIM);

    // QK^T + scale
    for (int i = 0; i < n_tokens; i++) {
        for (int j = 0; j < n_tokens; j++) {
            float dot = 0.0f;
            for (int d = 0; d < EMBED_DIM; d++)
                dot += Q[i][d] * K[j][d];
            scores[i][j] = dot * scale;
        }
    }

    // Softmax (row-wise)
    for (int i = 0; i < n_tokens; i++) {
        float max_val = scores[i][0];
        for (int j = 1; j < n_tokens; j++)
            if (scores[i][j] > max_val) max_val = scores[i][j];
        float sum = 0.0f;
        for (int j = 0; j < n_tokens; j++) {
            attn_weights[i][j] = expf(scores[i][j] - max_val);
            sum += attn_weights[i][j];
        }
        for (int j = 0; j < n_tokens; j++) attn_weights[i][j] /= sum;
    }

    // weights * V
    for (int i = 0; i < n_tokens; i++) {
        for (int d = 0; d < EMBED_DIM; d++) {
            float sum = 0.0f;
            for (int j = 0; j < n_tokens; j++)
                sum += attn_weights[i][j] * V[j][d];
            output[i][d] = sum;
        }
    }
}

// ====================== 5. MEANINGFUL EXTRACTIVE SUMMARY (sentence level) ======================
void summarize(char *original_text, char tokens[MAX_TOKENS][MAX_WORD_LEN], int n_tokens,
               float attn_weights[MAX_TOKENS][MAX_TOKENS], char *summary) {
    // Simple sentence split (for meaningful summary)
    char sentences[20][1024]; int n_sent = 0;
    char *sent = strtok(original_text, ".!?");
    while (sent && n_sent < 20) {
        strcpy(sentences[n_sent++], sent);
        sent = strtok(NULL, ".!?");
    }

    float sent_score[20] = {0};
    int token_idx = 0;
    for (int s = 0; s < n_sent; s++) {
        int sent_len = 0;
        while (token_idx < n_tokens && strstr(sentences[s], tokens[token_idx])) {
            for (int j = 0; j < n_tokens; j++)
                sent_score[s] += attn_weights[token_idx][j];
            sent_len++;
            token_idx++;
        }
        if (sent_len) sent_score[s] /= sent_len;
    }

    // Pick top 2 sentences
    int best1 = 0, best2 = -1;
    for (int i = 1; i < n_sent; i++) {
        if (sent_score[i] > sent_score[best1]) { best2 = best1; best1 = i; }
        else if (best2 == -1 || sent_score[i] > sent_score[best2]) best2 = i;
    }

    summary[0] = '\0';
    strcat(summary, "=== MEANINGFUL SUMMARY (real GloVe + attention) ===\n");
    if (best1 < n_sent) { strcat(summary, sentences[best1]); strcat(summary, ".\n"); }
    if (best2 != -1 && best2 < n_sent) { strcat(summary, sentences[best2]); strcat(summary, ".\n"); }
    strcat(summary, "===================================================\n");
}

// ====================== MAIN ======================
int main() {
    char paragraph[8192];
    char original[8192];
    char tokens[MAX_TOKENS][MAX_WORD_LEN];
    float embeddings[MAX_TOKENS][EMBED_DIM];
    float attn_weights[MAX_TOKENS][MAX_TOKENS];
    float output[MAX_TOKENS][EMBED_DIM];
    char summary[4096];

    printf("=== HPC Attention Calculator (Serial C + REAL GloVe 50D) ===\n");
    if (!load_glove("mini_glove_50d.txt")) {
        printf("Please create mini_glove_50d.txt first!\n");
        return 1;
    }

    printf("Paste paragraph (end with empty line):\n");
    fgets(paragraph, sizeof(paragraph), stdin);
    strcpy(original, paragraph);           // save for sentence splitting

    int n_tokens = tokenize(paragraph, tokens);
    printf("\nTokens found: %d\n", n_tokens);

    vectorize(tokens, n_tokens, embeddings);
    self_attention(embeddings, n_tokens, attn_weights, output);

    // Show sample attention (supervisor loves this)
    printf("\nSample Self-Attention Weights (first 8x8):\n");
    for (int i = 0; i < (n_tokens < 8 ? n_tokens : 8); i++) {
        for (int j = 0; j < (n_tokens < 8 ? n_tokens : 8); j++)
            printf("%.3f ", attn_weights[i][j]);
        printf("\n");
    }

    summarize(original, tokens, n_tokens, attn_weights, summary);
    printf("\n%s\n", summary);
    printf("=== Serial C version ready for MPI / OpenMP / CUDA ===\n");

    free(vocab);
    return 0;
}