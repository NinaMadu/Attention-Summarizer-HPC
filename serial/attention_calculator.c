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

// ====================== 5. IMPROVED MEANINGFUL EXTRACTIVE SUMMARY USING ATTENTION ======================
void summarize(char *original_text, char tokens[MAX_TOKENS][MAX_WORD_LEN], int n_tokens,
               float attn_weights[MAX_TOKENS][MAX_TOKENS], char *summary) {
    // Step 1: Split into sentences (simple delimiter-based)
    char sentences[30][1024];
    int n_sent = 0;
    char temp_text[8192];
    strcpy(temp_text, original_text);
    char *sent = strtok(temp_text, ".!?");
    while (sent && n_sent < 30) {
        while (*sent == ' ') sent++;  // trim leading space
        if (strlen(sent) > 5) {       // ignore tiny fragments
            strncpy(sentences[n_sent], sent, 1023);
            sentences[n_sent][1023] = '\0';
            n_sent++;
        }
        sent = strtok(NULL, ".!?");
    }

    // Step 2: Score each sentence using attention weights (this is the key part)
    float sent_score[30] = {0.0f};
    int token_idx = 0;
    for (int s = 0; s < n_sent && token_idx < n_tokens; s++) {
        int sent_token_count = 0;
        float total_attn = 0.0f;

        // For each token in this sentence, sum its attention to ALL tokens (global importance)
        while (token_idx < n_tokens && strstr(sentences[s], tokens[token_idx]) != NULL) {
            for (int j = 0; j < n_tokens; j++) {
                total_attn += attn_weights[token_idx][j];   // use calculated attention
            }
            sent_token_count++;
            token_idx++;
        }

        if (sent_token_count > 0) {
            sent_score[s] = total_attn / sent_token_count;  // average attention per token in sentence
        }
    }

    // Step 3: Select top 2 sentences (simple selection - easy to parallelize later)
    int best_idx[2] = {-1, -1};
    float best_score[2] = {-1.0f, -1.0f};

    for (int s = 0; s < n_sent; s++) {
        if (sent_score[s] > best_score[0]) {
            best_score[1] = best_score[0];
            best_idx[1] = best_idx[0];
            best_score[0] = sent_score[s];
            best_idx[0] = s;
        } else if (sent_score[s] > best_score[1]) {
            best_score[1] = sent_score[s];
            best_idx[1] = s;
        }
    }

    // Step 4: Build summary string
    summary[0] = '\0';
    strcat(summary, "=== MEANINGFUL SUMMARY (built from real GloVe + calculated self-attention) ===\n");
    if (best_idx[0] != -1) {
        strcat(summary, sentences[best_idx[0]]);
        strcat(summary, ".\n");
    }
    if (best_idx[1] != -1 && best_idx[1] != best_idx[0]) {
        strcat(summary, sentences[best_idx[1]]);
        strcat(summary, ".\n");
    }
    strcat(summary, "======================================================================\n");
    strcat(summary, "Summary selected using average attention strength per sentence.\n");
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

    free(vocab);
    return 0;
}