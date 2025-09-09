//
// Created by play_ on 08/09/2025.
//

#ifndef PIETRO1_H
#define PIETRO1_H

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

#include "csv.h"
#include "log.h"
#include "timer.h"

typedef float REAL;

typedef struct {
    int num_layers;
    int *layer_sizes;
    REAL ****weights;
    REAL **biases;
    REAL **activations;
    REAL **deltas;
    REAL learning_rate;
    char **neuron_active;
} MLP;

#define SIGMOID(x)        ((REAL)(1.0 / (1.0 + exp(-(x)))))
#define SIGMOID_DER(x)    ((REAL)((x) * (1.0 - (x))))
#define RELU(x)           ((x) > 0 ? (x) : 0)
#define RELU_DER(x)       ((x) > 0 ? 1 : 0)
#define LEAKY_ALPHA       0.01
#define LEAKY_RELU(x)     ((x) > 0 ? (x) : LEAKY_ALPHA * (x))
#define LEAKY_RELU_DER(x) ((x) > 0 ? 1.0 : LEAKY_ALPHA)
#define RAND_WEIGHT()     ((REAL)(((REAL)rand() / RAND_MAX) * 2.0 - 1.0))
#define NUM_FEATURES      35.0
#define NUM_CLASSES       9.0
#define MAX_NUMBER        (+sqrt(1/NUM_FEATURES))
#define MIN_NUMBER        (-sqrt(1/NUM_FEATURES))
#define RAND_PYTORCH()    ((REAL)((MAX_NUMBER - MIN_NUMBER)*((REAL)rand() / RAND_MAX) + MIN_NUMBER))
#define LIMIT_PLUS        +sqrt(6.0 / (NUM_FEATURES + NUM_CLASSES))
#define LIMIT_MINUS       -sqrt(6.0 / (NUM_FEATURES + NUM_CLASSES))
#define RAND_XAVIER()     ((REAL)(((REAL)rand() / RAND_MAX) * (LIMIT_PLUS - LIMIT_MINUS) + LIMIT_MINUS))
#define RAND_UNIFORM()    ((REAL)rand() / (REAL)RAND_MAX)
#define RAND_HE()         ( ((RAND_UNIFORM() * 2.0) - 1.0) * sqrt(6.0 / (NUM_FEATURES)) )
#define DEBUG_BACKWARD    0

#define BASE_DIR "/pietro"


inline bool read_next_sample(File &f, int x_cols, int y_cols, REAL *x, REAL *y) {
    int total = x_cols + y_cols;
    float tmp[64];
    if (total > (int) (sizeof(tmp) / sizeof(tmp[0]))) return false;
    if (!csv_read(tmp, total, f)) return false;
    for (int i = 0; i < x_cols; ++i) x[i] = (REAL) tmp[i];
    for (int j = 0; j < y_cols; ++j) y[j] = (REAL) tmp[x_cols + j];
    return true;
}

inline MLP *create_mlp(int num_layers,
                       int *layer_sizes,
                       REAL lr) {
    MLP *mlp = (MLP *) malloc(sizeof(MLP));
    mlp->num_layers = num_layers;
    mlp->learning_rate = lr;
    mlp->layer_sizes = (int *) malloc(num_layers * sizeof(int));
    for (int i = 0; i < num_layers; i++) {
        mlp->layer_sizes[i] = layer_sizes[i];
    }
    mlp->activations = (REAL **) malloc(num_layers * sizeof(REAL *));
    mlp->deltas = (REAL **) malloc(num_layers * sizeof(REAL *));
    mlp->neuron_active = (char **) malloc(num_layers * sizeof(char *));
    for (int i = 0; i < num_layers; i++) {
        mlp->activations[i] = (REAL *) calloc(layer_sizes[i], sizeof(REAL));
        mlp->deltas[i] = (REAL *) calloc(layer_sizes[i], sizeof(REAL));
        mlp->neuron_active[i] = (char *) malloc(layer_sizes[i] * sizeof(char));
        for (int j = 0; j < layer_sizes[i]; j++) {
            mlp->neuron_active[i][j] = 1;
        }
    }
    mlp->weights = (REAL ****) malloc((num_layers - 1) * sizeof(REAL ***));
    mlp->biases = (REAL **) malloc((num_layers - 1) * sizeof(REAL *));
    for (int l = 1; l < num_layers; l++) {
        mlp->biases[l - 1] = (REAL *) malloc(layer_sizes[l] * sizeof(REAL));
        mlp->weights[l - 1] = (REAL ***) malloc(layer_sizes[l] * sizeof(REAL **));
        for (int n = 0; n < layer_sizes[l]; n++) {
            mlp->biases[l - 1][n] = RAND_PYTORCH();
            mlp->weights[l - 1][n] = (REAL **) malloc(layer_sizes[l - 1] * sizeof(REAL *));
            for (int w = 0; w < layer_sizes[l - 1]; w++) {
                mlp->weights[l - 1][n][w] = (REAL *) malloc(sizeof(REAL));
                *mlp->weights[l - 1][n][w] = RAND_PYTORCH();
            }
        }
    }
    return mlp;
}

inline void free_mlp(MLP *mlp) {
    if (!mlp) return;
    for (int l = 1; l < mlp->num_layers; l++) {
        for (int n = 0; n < mlp->layer_sizes[l]; n++) {
            for (int w = 0; w < mlp->layer_sizes[l - 1]; w++)
                free(mlp->weights[l - 1][n][w]);
            free(mlp->weights[l - 1][n]);
        }
        free(mlp->weights[l - 1]);
        free(mlp->biases[l - 1]);
    }
    free(mlp->weights);
    free(mlp->biases);
    for (int i = 0; i < mlp->num_layers; i++) {
        free(mlp->activations[i]);
        free(mlp->deltas[i]);
        free(mlp->neuron_active[i]);
    }
    free(mlp->activations);
    free(mlp->deltas);
    free(mlp->neuron_active);
    free(mlp->layer_sizes);
    free(mlp);
}

inline void forward_pass(MLP *mlp, REAL *input) {
    for (int i = 0; i < mlp->layer_sizes[0]; i++) {
        mlp->activations[0][i] = input[i];
    }
    for (int l = 1; l < mlp->num_layers; l++) {
        for (int n = 0; n < mlp->layer_sizes[l]; n++) {
            if (!mlp->neuron_active[l][n]) {
                continue;
            }
            REAL sum = mlp->biases[l - 1][n];
            for (int p = 0; p < mlp->layer_sizes[l - 1]; p++) {
                if (mlp->neuron_active[l - 1][p] && mlp->weights[l - 1][n][p]) {
                    sum += mlp->activations[l - 1][p] * (*mlp->weights[l - 1][n][p]);
                }
            }
            if (l == mlp->num_layers - 1) {
                mlp->activations[l][n] = sum;
            } else {
                mlp->activations[l][n] = LEAKY_RELU(sum);
            }
        }
    }
}

inline void backward_pass(MLP *mlp,
                          REAL *target) {
    int out = mlp->num_layers - 1;
    for (int n = 0; n < mlp->layer_sizes[out]; n++) {
        if (!mlp->neuron_active[out][n]) {
            continue;
        }
        REAL output = mlp->activations[out][n];
        mlp->deltas[out][n] = (output - target[n]);
    }
    for (int l = out - 1; l >= 1; l--) {
        for (int n = 0; n < mlp->layer_sizes[l]; n++) {
            if (!mlp->neuron_active[l][n]) {
                continue;
            }
            REAL error = 0.0;
            for (int nn = 0; nn < mlp->layer_sizes[l + 1]; nn++) {
                if (!(mlp->neuron_active[l + 1][nn] &&
                      mlp->weights[l][nn] &&
                      mlp->weights[l][nn][n])) {
#if DEBUG_BACKWARD
                    printf("[DEBUG] Skipped weight L%d->L%d[%d]\n", l, l+1, nn);
#endif
                    continue;
                }
                error += mlp->deltas[l + 1][nn] * (*mlp->weights[l][nn][n]);
            }
            mlp->deltas[l][n] = error * LEAKY_RELU_DER(mlp->activations[l][n]);
        }
    }
    for (int l = 1; l < mlp->num_layers; l++) {
        for (int n = 0; n < mlp->layer_sizes[l]; n++) {
            if (!mlp->neuron_active[l][n]) {
                continue;
            }
            mlp->biases[l - 1][n] -= mlp->learning_rate * mlp->deltas[l][n];
            if (!mlp->weights[l - 1][n]) {
                continue;
            }
            for (int p = 0; p < mlp->layer_sizes[l - 1]; p++) {
                if (!(mlp->neuron_active[l - 1][p] && mlp->weights[l - 1][n][p])) {
#if DEBUG_BACKWARD
                    printf("[DEBUG] Skipped update weight L%d[%d]->L%d[%d]\n", l-1, p, l, n);
#endif
                    continue;
                }
                *mlp->weights[l - 1][n][p] -= mlp->learning_rate *
                        mlp->deltas[l][n] *
                        mlp->activations[l - 1][p];
            }
        }
    }
}

inline void one_hot(int label,
                    int n_classes,
                    REAL *output) {
    for (int i = 0; i < n_classes; i++) {
        output[i] = 0.0;
    }
    if (label >= 0 && label < n_classes) {
        output[label] = 1.0;
    }
}

inline void softmax(const REAL *logits,
                    REAL *output,
                    int n_classes) {
    REAL max_val = logits[0];
    for (int i = 1; i < n_classes; i++) {
        if (logits[i] > max_val) {
            max_val = logits[i];
        }
    }
    REAL sum_exp = 0.0;
    for (int i = 0; i < n_classes; i++) {
        output[i] = exp(logits[i] - max_val);
        sum_exp += output[i];
    }
    for (int i = 0; i < n_classes; i++) {
        output[i] /= sum_exp;
    }
}

inline REAL cross_entropy_loss(const REAL *pred,
                               const REAL *target,
                               int num_classes) {
    REAL loss = 0.0;
    const REAL eps = 1e-12;
    for (int i = 0; i < num_classes; i++) {
        if (target[i] > 0.5) {
            loss = -log(pred[i] + eps);
            break;
        }
    }
    return loss;
}

inline REAL mse_loss(const REAL *logits,
                     const REAL *y_true,
                     int n_classes) {
    REAL y_pred[n_classes];
    softmax(logits, y_pred, n_classes);
    REAL loss = 0.0;
    for (int i = 0; i < n_classes; i++) {
        REAL diff = y_pred[i] - y_true[i];
        loss += diff * diff;
    }
    return loss / n_classes;
}

inline int argmax(const REAL *array,
                  int n) {
    int max_index = 0;
    REAL max_val = array[0];
    for (int i = 1; i < n; i++) {
        if (array[i] > max_val) {
            max_val = array[i];
            max_index = i;
        }
    }
    return max_index;
}

inline REAL accuracy(int true_label,
                     const REAL *y_pred,
                     int n_classes) {
    int pred_label = argmax(y_pred, n_classes);
    return (pred_label == true_label) ? 1.0 : 0.0;
}

inline int count_active_weights(MLP *mlp) {
    int count = 0;
    for (int l = 1; l < mlp->num_layers; l++) {
        for (int n = 0; n < mlp->layer_sizes[l]; n++) {
            if (!mlp->neuron_active[l][n]) {
                continue;
            }
            for (int p = 0; p < mlp->layer_sizes[l - 1]; p++) {
                if (mlp->neuron_active[l - 1][p] && mlp->weights[l - 1][n][p]) {
                    count++;
                }
            }
        }
    }
    return count;
}

inline void evaluate_stream(File &test_f, int x_cols, int y_cols, int samples,
                            REAL *test_loss, REAL *test_accuracy, MLP *mlp) {
    int num_classes = mlp->layer_sizes[mlp->num_layers - 1];

    REAL probs[num_classes];
    REAL onehot[num_classes];
    REAL x[x_cols];
    REAL y[y_cols];

    REAL loss_val = 0.0;
    int correct = 0;

    for (int i = 0; i < samples; ++i) {
        if (!read_next_sample(test_f, x_cols, y_cols, x, y)) break;
        forward_pass(mlp, x);
        softmax(mlp->activations[mlp->num_layers - 1], probs, num_classes);
        one_hot((int) y[0], num_classes, onehot);
        loss_val += mse_loss(mlp->activations[mlp->num_layers - 1], onehot, num_classes);
        correct += accuracy((int) y[0], probs, num_classes);
    }

    *test_loss = loss_val / samples;
    *test_accuracy = (REAL) correct / samples;
}

typedef struct {
    int layer;
    int neuron;
    int prev;
    REAL absval; // |weight|
} PruneCand;

static inline void prune_weight(MLP *mlp, int layer, int neuron, int prev) {
    if (!mlp || layer <= 0 || layer >= mlp->num_layers) return;
    if (!mlp->weights[layer - 1][neuron][prev]) return;
    free(mlp->weights[layer - 1][neuron][prev]);
    mlp->weights[layer - 1][neuron][prev] = NULL;
}

/* --- max-heap su absval, grandezza massima = K --- */
static inline void heap_swap(PruneCand *h, int i, int j) {
    PruneCand t = h[i];
    h[i] = h[j];
    h[j] = t;
}

static inline void heap_sift_up(PruneCand *h, int i) {
    while (i > 0) {
        int p = (i - 1) / 2;
        if (h[p].absval >= h[i].absval) break;
        heap_swap(h, p, i);
        i = p;
    }
}

static inline void heap_sift_down(PruneCand *h, int n, int i) {
    for (;;) {
        int l = 2 * i + 1, r = l + 1, m = i;
        if (l < n && h[l].absval > h[m].absval) m = l;
        if (r < n && h[r].absval > h[m].absval) m = r;
        if (m == i) break;
        heap_swap(h, i, m);
        i = m;
    }
}

static inline void heap_try_push(PruneCand *h, int *sz, int cap, PruneCand x) {
    if (*sz < cap) {
        h[(*sz)++] = x;
        heap_sift_up(h, *sz - 1);
    } else if (cap > 0 && x.absval < h[0].absval) {
        h[0] = x;
        heap_sift_down(h, cap, 0);
    }
}

/* --- pruning per percentuale con memoria O(K) --- */
void prune_by_percentage(MLP *mlp, REAL prune_percent) {
    if (!mlp || prune_percent <= 0.0 || prune_percent >= 100.0) return;

    /* Passata 1: conta i pesi presenti */
    int total = 0;
    for (int l = 0; l < mlp->num_layers - 1; l++) {
        for (int n = 0; n < mlp->layer_sizes[l + 1]; n++) {
            for (int p = 0; p < mlp->layer_sizes[l]; p++) {
                if (mlp->weights[l][n][p]) total++;
            }
        }
    }
    if (total == 0) return;

    int K = (int) ((double) total * (double) prune_percent / 100.0);
    if (K <= 0) return;

    PruneCand *heap = (PruneCand *) malloc(sizeof(PruneCand) * (size_t) K);
    if (!heap) return; // niente memoria, esci pulito
    int hsz = 0;

    /* Passata 2: mantieni i K più piccoli in valore assoluto */
    for (int l = 0; l < mlp->num_layers - 1; l++) {
        for (int n = 0; n < mlp->layer_sizes[l + 1]; n++) {
            for (int p = 0; p < mlp->layer_sizes[l]; p++) {
                REAL *wptr = mlp->weights[l][n][p];
                if (!wptr) continue;
                PruneCand c = {.layer = l + 1, .neuron = n, .prev = p, .absval = fabs(*wptr)};
                heap_try_push(heap, &hsz, K, c);
            }
        }
    }

    /* Il max-heap contiene i K candidati più “piccoli” da potare, in ordine arbitrario */
    for (int i = 0; i < hsz; i++) {
        prune_weight(mlp, heap[i].layer, heap[i].neuron, heap[i].prev);
    }

    LOG_INFO("Pruned %d/%d pesi (%.1f%%)\n", hsz, total, prune_percent);
    free(heap);
}


inline void train_with_pruning_stream(MLP *mlp,
                                      int x_cols,
                                      int y_cols,
                                      int samples,
                                      int epochs,
                                      int    test_samples,
                                      REAL   prune_total_percent,   // % totale di pruning sugli iniziali
                                      int    num_prune_steps,       // quante sessioni di pruning
                                      REAL   lr_decay,
                                      int    log_freq ) {
    REAL epoch_loss = 0;
    REAL epoch_accuracy = 0;
    REAL test_loss = 0.0f;
    REAL test_accuracy = 0.0f;
    int correct = 0;
    REAL initial_lr = mlp->learning_rate;
    int num_classes = mlp->layer_sizes[mlp->num_layers - 1];

    REAL output_probs[num_classes];
    REAL one_hot_target[num_classes];
    REAL x[x_cols];
    REAL y[y_cols];

    REAL warmup_factor = 1.5;
    int warmup_epochs = 3;
    int last_prune_epoch = -1;

    Timer t = {0};


    LOG_INFO("epoch,epoch_loss,epoch_accuracy,active_weights,pruned_percent\n");
    int initial_weights = count_active_weights(mlp);

    File train_f, test_f;
    if (!open_csv(train_f, BASE_DIR, "Train_Data.csv", FILE_READ) ||
        !open_csv(test_f, BASE_DIR, "Test_Data.csv", FILE_READ)) {
        FAILURE("Errore apertura file CSV");
    }

    REAL prune_step_percent   = prune_total_percent / num_prune_steps;
    int prune_schedule[num_prune_steps];
    for (int i = 0; i < num_prune_steps; i++)
    {
        prune_schedule[i] = (int)((double)(i+1) * epochs / (num_prune_steps+1));
    }
    int  next_prune_idx = 0;


    for (int epoch = 0; epoch < epochs; epoch++) {
        epoch_loss = 0;
        epoch_accuracy = 0;
        correct = 0;
        mlp->learning_rate = initial_lr * pow(lr_decay, epoch);

        if (last_prune_epoch >= 0) {
            int since_prune = epoch - last_prune_epoch;
            if (since_prune >= 0 && since_prune < warmup_epochs) {
                mlp->learning_rate *= warmup_factor;
            }
        }

        START_TIMER(t);

        for (int sample = 0; sample < samples; sample++) {
            if (!read_next_sample(train_f, x_cols, y_cols, x, y)) break;

            forward_pass(mlp, x);
            softmax(mlp->activations[mlp->num_layers - 1], output_probs, num_classes);
            one_hot((int) y[0], num_classes, one_hot_target);

            //epoch_loss += mse_loss( mlp->activations[mlp->num_layers - 1], one_hot_target, num_classes );
            epoch_loss += cross_entropy_loss( output_probs, one_hot_target, num_classes );
            correct += accuracy((int) y[0], output_probs, num_classes);

            backward_pass(mlp, one_hot_target);
        }

        STOP_TIMER(t, "Stop Training");
        vTaskDelay(pdMS_TO_TICKS(5000));


        epoch_accuracy = (REAL) correct / samples;
        epoch_loss /= samples;

        /*PUNING*/
        if (next_prune_idx < num_prune_steps && epoch == prune_schedule[next_prune_idx])
        {
            REAL target_total_prune = (next_prune_idx + 1) * prune_step_percent;
            int  target_pruned_w    = (int)(initial_weights * target_total_prune / 100.0);
            int  active             = count_active_weights(mlp);
            int  already_pruned     = initial_weights - active;
            int  to_prune_now       = target_pruned_w - already_pruned;

            if (to_prune_now > 0)
            {
                REAL step_percent = 100.0 * to_prune_now / active;
                prune_by_percentage(mlp, step_percent);
            }

            last_prune_epoch = epoch;
            next_prune_idx++;
        }


        if (epoch % log_freq == 0) {
            int active = count_active_weights(mlp);
            REAL pruned_pct = 100.0 * (initial_weights - active) / initial_weights;
            size_t used_bytes = 0;
            LOG_INFO("%d,%.6f,%.6f,%d,%.2f\n", epoch, epoch_loss, epoch_accuracy, active, pruned_pct);
            LOG_INFO(
                "Epoch %3d/%d: LOSS = %.6f ACCURACY = %.6f | ACTIVE WEIGHTS = %d (%.1f%% pruned) | MEMORY USED: %zu bytes (~%.2f MB) | LR = %.6f\n",
                (epoch+1), epochs, epoch_loss, epoch_accuracy, active, pruned_pct, used_bytes,
                used_bytes / (1024.0 * 1024.0), mlp->learning_rate);
        }

        START_TIMER(t);
        if (test_f) {
            evaluate_stream(test_f, x_cols, y_cols, test_samples, &test_loss, &test_accuracy, mlp);
            LOG_INFO("epoch,loss,accuracy\n");
            LOG_INFO("%d,%.6f,%.4f\n", epochs, test_loss, test_accuracy);
        }
        STOP_TIMER(t, "Stop Testing");
        RESET_ALL_FILES(train_f, test_f);
    }
}


#endif //PIETRO1_H
