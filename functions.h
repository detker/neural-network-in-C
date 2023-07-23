void load_data(double **digits, double **labels, 
                bool training_data);

void preprocess_digits_data(double **digits, 
                                bool training_data);

void to_categorical(double **labels, double **y, 
                        bool training_data);

double random_number();

double **allocate_array(const int n, const int m);

void free_array(double **arr);

double **generage_weights(const int n, const int m);

double **create_batch(double **input, const int batch_start, 
                        const int batch_end, const int batch_size,
                        const int digit_size);

double **matrix_transpose(double **A, const int n, const int m);

double **matrix_mul(double **A, const int n_A, const int m_A,
                    double **B, const int n_B, const int m_B);

double compute_error(double **y_hat, double **y, 
                        int batch_start, int batch_end,
                        const int classes);

bool is_prediction_correct(double **y_hat, double **y,
                            const int classes, int data_point_hat, 
                            int data_point);

double **compute_delta(double **layer_2, double **y,
                        int batch_start, int batch_end,
                        int classes);

void delta_divide(double **array, const int n, const int m, int divider);

void mull_w(double **matrix, const int n, const int m, double scalar);

void subtract_matrices(double **A, double **B, const int n, const int m);

void relu(double **layer, const int n, const int m);

void relu2deriv(double **matrix, double **template, const int n, 
                const int m);

double **create_dropout(double dropout_val, const int n, const int m);

void inject_dropout(double **arr, double **dropout_arr, const int n,
                        const int m);

void print_head();