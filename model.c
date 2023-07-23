/*
MNIST digits recognition - neural network made entirely in C lang.
Author: @detker
Date: 20.07.2023
*/


// Importing necessary standard libraries + header file.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <assert.h>
#include <time.h>
#include <unistd.h>
#include "functions.h"


// Defining constants (including hyperparameters).
#define EPOCHS 300
#define BATCH_SIZE 100
#define HIDDEN_LAYER 100
#define ALPHA 0.001
#define DROPOUT_VAL 0.5
// ==================== //
#define N_TRAIN_DATA 800
#define N_VALID_DATA 200
#define DIGIT_SIZE 784
#define LINE_LENGTH 4206
#define OUT_CLASSES 10


int main(void)
{
    // Hello! :)
    print_head();

    // Defining training and testing data arrays.
    double **X_train = allocate_array(N_TRAIN_DATA, DIGIT_SIZE);
    double **labels_train = allocate_array(1, N_TRAIN_DATA);
    double **y_train = allocate_array(N_TRAIN_DATA, OUT_CLASSES);
    double **X_valid = allocate_array(N_VALID_DATA, DIGIT_SIZE);
    double **labels_valid = allocate_array(1, N_VALID_DATA);
    double **y_valid = allocate_array(N_VALID_DATA, OUT_CLASSES);

    // Loading raw .csv data into arrays.
    load_data(X_train, labels_train, true);
    load_data(X_valid, labels_valid, false);
    
    // Preprocessing, normalizing pixel values to be in <0, 1> 
    // range (inclusive).
    preprocess_digits_data(X_train, true);
    preprocess_digits_data(X_valid, false);
    
    // Categorizing labels (1-of-N), to match model's 10 output's.
    to_categorical(labels_train, y_train, true);
    to_categorical(labels_valid, y_valid, false);

    // Generating random weights values and allocating them in arrays.
    double **weights_0_1 = generage_weights(DIGIT_SIZE, HIDDEN_LAYER);
    double **weights_1_2 = generage_weights(HIDDEN_LAYER, OUT_CLASSES);
    
    // MODEL ARCHITECTURE & TRAINING.
    int batch_start, batch_end;
    double **layer_0, **layer_1, **layer_2;
    double **dropout_layer_1;
    double **delta_2, **delta_1;
    double **weights_1_2_T, **layer_1_T, **layer_0_T;
    double **weights_delta_1_2, **weights_delta_0_1;
    double error, correct_cnt, valid_error, valid_correct_cnt;

    for(int epoch=0; epoch<EPOCHS; ++epoch)
    {
        error = 0.0;
        correct_cnt = 0.0;
        for(int i=0; i<N_TRAIN_DATA/BATCH_SIZE; ++i)
        {
            batch_start = i*BATCH_SIZE;
            batch_end = (i+1)*BATCH_SIZE;
            
            // Computing layers.
            
            // SHAPE: (BATCH_SIZE, DIGIT_SIZE)
            layer_0 = create_batch(X_train, batch_start, 
                                    batch_end, BATCH_SIZE,
                                    DIGIT_SIZE);
            
            // SHAPE: (BATCH_SIZE, HIDDEN_LAYER) 
            layer_1 = matrix_mul(layer_0, BATCH_SIZE, DIGIT_SIZE, 
                                    weights_0_1, DIGIT_SIZE, HIDDEN_LAYER);
            // Adding ReLU as an activation function for layer_1.
            relu(layer_1, BATCH_SIZE, HIDDEN_LAYER);
            // Creating dropout layer to minimalize model overfitting.
            dropout_layer_1 = create_dropout(DROPOUT_VAL, BATCH_SIZE, HIDDEN_LAYER);
            // Applying dropout layer onto layer_1.
            inject_dropout(layer_1, dropout_layer_1, BATCH_SIZE, HIDDEN_LAYER);
            // Normalizing layer_1 output by multiplying all still-active outputs
            // by DROPOUT_VAL inversion.
            mull_w(layer_1, BATCH_SIZE, HIDDEN_LAYER, (double)1.0/DROPOUT_VAL);
            
            // SHAPE: (BATCH_SIZE, OUT_CLASSES)
            layer_2 = matrix_mul(layer_1, BATCH_SIZE, HIDDEN_LAYER,
                                    weights_1_2, HIDDEN_LAYER, OUT_CLASSES);
            
            
            // Computing batch's error.
            error += compute_error(layer_2, y_train, batch_start, batch_end,
                                    OUT_CLASSES);
            
            for(int k=0; k<BATCH_SIZE; ++k)
            {
                // Computing accuracy for every data point in the batch.
                correct_cnt += is_prediction_correct(layer_2, y_train, OUT_CLASSES,
                                                        k, batch_start+k);
                
                // Computing raw error between layer_2 and y_labels in a batch.
                // SHAPE: (BATCH_SIZE, OUT_CLASSES)
                delta_2 = compute_delta(layer_2, y_train, batch_start,
                                        batch_end, OUT_CLASSES);
                // Normalizing delta_2.
                delta_divide(delta_2, BATCH_SIZE, OUT_CLASSES, BATCH_SIZE);
                
                // Transposing weights matrix in order to perform matrix multiplication.
                // SHAPE: (OUT_CLASSES, HIDDEN_LAYER)
                weights_1_2_T = matrix_transpose(weights_1_2, HIDDEN_LAYER, OUT_CLASSES);                
                
                // Back-propagate error from layer_2 to layer_1.
                // SHAPE: (BATCH_SIZE, HIDDEN_LAYER)
                delta_1 = matrix_mul(delta_2, BATCH_SIZE, OUT_CLASSES,
                                        weights_1_2_T, OUT_CLASSES, HIDDEN_LAYER);
                // Applying derative of relu activation function in order to not modify neurons
                // that have no impact on the error.
                relu2deriv(delta_1, layer_1, BATCH_SIZE, HIDDEN_LAYER);
                // Applying dropout for the same reason.
                inject_dropout(delta_1, dropout_layer_1, BATCH_SIZE, HIDDEN_LAYER);


                // SHAPE: (HIDDEN_LAYER, BATCH_SIZE)
                layer_1_T = matrix_transpose(layer_1, BATCH_SIZE, HIDDEN_LAYER);
                // SHAPE: (DIGIT_SIZE, BATCH_SIZE)
                layer_0_T = matrix_transpose(layer_0, BATCH_SIZE, DIGIT_SIZE);                

                // Computing gradients by using partial deratives of loss function.
                // SHAPE: (HIDDEN_LAYER, OUT_CLASSES)
                weights_delta_1_2 = matrix_mul(layer_1_T, HIDDEN_LAYER, BATCH_SIZE,
                                                delta_2, BATCH_SIZE, OUT_CLASSES);
                // SHAPE: (DIGIT_SIZE, HIDDEN_LAYER)
                weights_delta_0_1 = matrix_mul(layer_0_T, DIGIT_SIZE, BATCH_SIZE,
                                                delta_1, BATCH_SIZE, HIDDEN_LAYER);


                // Multiplying deratives by alpha constant a.k.a. learning rate
                // in order to acquire precision in searching for global minimum
                // of loss function.
                mull_w(weights_delta_1_2, HIDDEN_LAYER, OUT_CLASSES, ALPHA);
                mull_w(weights_delta_0_1, DIGIT_SIZE, HIDDEN_LAYER, ALPHA);
                
                // Updating weights.
                subtract_matrices(weights_1_2, weights_delta_1_2, HIDDEN_LAYER,
                                    OUT_CLASSES);
                subtract_matrices(weights_0_1, weights_delta_0_1, DIGIT_SIZE,
                                    HIDDEN_LAYER);
                
                
                free_array(delta_2);
                free_array(weights_1_2_T);
                free_array(delta_1);
                free_array(layer_1_T);
                free_array(layer_0_T);
                free_array(weights_delta_1_2);
                free_array(weights_delta_0_1);

            }

            free_array(layer_0);
            free_array(layer_1);
            free_array(layer_2);
            free(dropout_layer_1);

        }

        // Every 10-th epoch model predicts results based on actual weights.
        if(epoch%10==0)
        {
            valid_error = 0.0;
            valid_correct_cnt = 0.0;
            for(int i=0; i<N_VALID_DATA; ++i)
            {
                // SHAPE: (1, DIGIT_SIZE)
                layer_0 = create_batch(X_valid, i, i+1, 
                                        1, DIGIT_SIZE);
                
                // SHAPE: (1, HIDDEN_LAYER)
                layer_1 = matrix_mul(layer_0, 1, DIGIT_SIZE, weights_0_1,
                                        DIGIT_SIZE, HIDDEN_LAYER);
                relu(layer_1, 1, HIDDEN_LAYER);
                
                // SHAPE: (1, OUT_CLASSES)
                layer_2 = matrix_mul(layer_1, 1, HIDDEN_LAYER, weights_1_2,
                                    HIDDEN_LAYER, OUT_CLASSES);
                
                valid_error += compute_error(layer_2, y_valid, i, i+1, OUT_CLASSES);
                valid_correct_cnt += is_prediction_correct(layer_2, y_valid, OUT_CLASSES,
                                                                0, i);

                free_array(layer_0);
                free_array(layer_1);
                free_array(layer_2);
            }

            printf("EPOCH: %d; ", epoch);
            printf("VALID_ERROR: %lf; ", valid_error/N_VALID_DATA);
            printf("VALID_ACCURACY: %lf; ", valid_correct_cnt/N_VALID_DATA);
            printf("TRAIN_ERROR: %lf; ", error/N_TRAIN_DATA);
            printf("TRAIN_ACC: %lf\n", correct_cnt/N_TRAIN_DATA);

        }
        else
        {
            printf("EPOCH: %d; ", epoch);
            printf("TRAIN_ERROR: %lf; ", error/N_TRAIN_DATA);
            printf("TRAIN_ACC: %lf\n", correct_cnt/N_TRAIN_DATA);
        }
    }

    free_array(weights_0_1);
    free_array(weights_1_2);

    free_array(X_train);
    free_array(labels_train);
    free_array(y_train);
    free_array(X_valid);
    free_array(labels_valid);
    free_array(y_valid);
    
    return 0;
}