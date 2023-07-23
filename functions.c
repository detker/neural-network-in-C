#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <assert.h>
#include <time.h>
#include "functions.h"

#define N_TRAIN_DATA 800
#define N_VALID_DATA 200
#define DIGIT_SIZE 784
#define LINE_LENGTH 4206
#define OUT_CLASSES 10

#ifdef _WINDOWS
#include <windows.h>
#else
#include <unistd.h>
#ifdef __APPLE__
#define Sleep(x) usleep((x)*1000)
#else
#define Sleep(x) sleep((x)/1000)
#endif
#endif

void load_data(double **digits, double **labels, 
                bool training_data) 
{
    char *source = "archive/mnist_train.csv";
    int n = N_TRAIN_DATA;
    if(!training_data)
    {
        n = N_VALID_DATA;
        source = "archive/mnist_test.csv";
    }

    const int N_DATA = n;
    char line[LINE_LENGTH];
    char *token;

    FILE *fp = fopen(source, "r");
    if(!fp) exit(0);

    int i = 0, j = 0;
    while((i<N_DATA) && (fgets(line, LINE_LENGTH, fp)))
    {
        j = 0;
        token = strtok(line, ",");
        labels[0][i] = (int)atoi(token);
        token = strtok(NULL, ",");
        while(token)
        {
            digits[i][j] = atof(token);
            ++j;
            token = strtok(NULL, ",");
        }
        ++i;
    }

    fclose(fp);
}

void preprocess_digits_data(double **digits, bool training_data)
{
    int n = N_TRAIN_DATA;
    if(!training_data) n = N_VALID_DATA;
    const int N_DATA = n;
    
    for(int i=0; i<N_DATA; ++i)
    {
        for(int j=0; j<DIGIT_SIZE; ++j)
            digits[i][j] = digits[i][j]/(double)255.0;
    }
}

void to_categorical(double **labels, double **y, 
                        bool training_data)
{
    int n = N_TRAIN_DATA;
    if(!training_data) n = N_VALID_DATA;
    const int N_DATA = n;

    for(int i=0; i<N_DATA; ++i)
    {
        for(int j=0; j<OUT_CLASSES; ++j)
        {
            if(j == (int)labels[0][i]) y[i][j] = 1;
            else y[i][j] = 0;
        }
    }
}

double **allocate_array(const int n, const int m)
{
    double *row = calloc((size_t)((n*m)+1), sizeof(&row));
    double **rows = malloc((size_t)(n+1)*sizeof(&rows));
    
    for(int i=0; i<n; ++i) rows[i] = row + i*m;
    
    return rows;
}

double **generage_weights(const int n, const int m)
{
    srand((unsigned int)time(NULL));
    double **weights = allocate_array(n, m);
    
    for(int i=0; i<n; ++i)
    {
        for(int j=0; j<m; ++j)
            weights[i][j] = 0.2*((double)rand()/(double)(RAND_MAX))-0.1;
    }
    
    return weights;
}

void free_array(double **arr)
{
    free(*arr);
    free(arr);
}

double **matrix_transpose(double **A, const int n, const int m)
{
    double **transposed = allocate_array(m, n);
    
    for(int i=0; i<n; ++i)
    {
        for(int j=0; j<m; ++j)
            transposed[j][i] = A[i][j];
    }
    
    return transposed;
}

double **create_batch(double **input, const int batch_start, 
                        const int batch_end, const int batch_size,
                        const int digit_size)
{
    double **layer_0 = allocate_array(batch_size, digit_size);
    
    for(int i=batch_start; i<batch_end; ++i)
    {
        for(int j=0; j<digit_size; ++j) 
            layer_0[i-batch_start][j] = input[i][j];
    }
    
    return layer_0;
}

double **matrix_mul(double **A, const int n_A, const int m_A,
                    double **B, const int n_B, const int m_B)
{
    assert(m_A == n_B);
    double **result = allocate_array(n_A, m_B);
    
    for(int i=0; i<m_B; ++i)
    {
        for(int j=0; j<n_A; ++j)
        {
            double sum = 0.0;
            for(int k=0; k<m_A; ++k)
            {
                sum += A[j][k] * B[k][i];
            }
            result[j][i] = sum;
        }
    }
    
    return result;
}

double compute_error(double **y_hat, double **y, 
                        int batch_start, int batch_end,
                        const int classes)
{
    double temp;
    double error = 0.0;
    
    for(int i=batch_start; i<batch_end; ++i)
    {
        for(int j=0; j<classes; ++j)
        {
            temp = (y_hat[i-batch_start][j]-y[i][j]);
            temp *= temp;
            error += temp;
        }
    }
    
    return error;
}

bool is_prediction_correct(double **y_hat, double **y,
                            const int classes, int data_point_hat, 
                            int data_point)
{
    double greatest_val = y_hat[data_point_hat][0];
    int greatest_val_index = 0;
    double one = y[data_point][0];
    int one_index = 0;

    for(int i=1; i<classes; ++i)
    {
        if(greatest_val < y_hat[data_point_hat][i])
        {
            greatest_val = y_hat[data_point_hat][i];
            greatest_val_index = i;
        }
        if(one < y[data_point][i])
        {
            one = y[data_point][i];
            one_index = i;
        }
    }

    return (greatest_val_index == one_index);
}

double **compute_delta(double **layer_2, double **y,
                        int batch_start, int batch_end,
                        int classes)
{
    double **delta = allocate_array(batch_end-batch_start+1, classes);
    
    for(int i=batch_start; i<batch_end; ++i)
    {
        for(int j=0; j<classes; ++j)
            delta[i-batch_start][j] = (layer_2[i-batch_start][j]-y[i][j]);
    }
    
    return delta;
}

void delta_divide(double **array, const int n, const int m, int divider)
{
    for(int i=0; i<n; ++i)
    {
        for(int j=0; j<m; ++j)
            array[i][j] /= (double)divider;
    }
}

void mull_w(double **matrix, const int n, const int m, double scalar)
{
    for(int i=0; i<n; ++i)
    {
        for(int j=0; j<m; ++j)
            matrix[i][j] *= scalar;
    }
}

void subtract_matrices(double **A, double **B, const int n, const int m)
{
    for(int i=0; i<n; ++i)
    {
        for(int j=0; j<m; ++j)
            A[i][j] -= B[i][j];
    }
}

void relu(double **layer, const int n, const int m)
{
    for(int i=0; i<n; ++i)
    {
        for(int j=0; j<m; ++j)
            layer[i][j] *= (layer[i][j]>0);
    }
}

void relu2deriv(double **matrix, double **template, const int n, 
                const int m)
{
    for(int i=0; i<n; ++i)
    {
        for(int j=0; j<m; ++j)
            matrix[i][j] *= (template[i][j]>0);
    }
}

double **create_dropout(double dropout_val, const int n, const int m)
{
    int to_eliminate = (int)(n * m * dropout_val);
    double **dropout = allocate_array(n, m);
    srand((unsigned int)time(NULL));

    for(int i=0; i<n; ++i)
    {
        for(int j=0; j<m; ++j)
            dropout[i][j] = 1.0;
    }

    int k = 0;
    int i=0, j=0;
    while(k < to_eliminate)
    {
        i = (int)rand() % n;
        j = (int)rand() % m;
        if((int)dropout[i][j] != 0)
        {
            dropout[i][j] = 0;
            k+=1;
        }
    }

    return dropout;
}

void inject_dropout(double **arr, double **dropout_arr, const int n,
                        const int m)
{
    for(int i=0; i<n; ++i)
    {
        for(int j=0; j<m; ++j)
            arr[i][j] *= dropout_arr[i][j];
    }
}

void print_head()
{
    printf("==============\n");
    printf("""(\\(\\\n(-.-)\no_('')(''))\n""");
    printf("==============\n");
    Sleep(1000);
    printf("C lang neural network implemented by: @detker.\n");
    printf("Training on: MNIST Dataset\n");
    Sleep(2000);
    printf("(yeah, it computes a little while...)\n");
}
