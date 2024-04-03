#include <iostream>
#include <mpi.h>

const int N = 4; // Размер матрицы и вектора

void printMatrix(double* A, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << A[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

void printVector(double* V, int size) {
    for (int i = 0; i < size; ++i) {
        std::cout << V[i] << " ";
    }
    std::cout << std::endl;
}

int main(int argc, char** argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int blockSize = N / size; // Размер блока для каждого процесса

    double* A = new double[N * N];
    double* b = new double[N];
    double* x = nullptr;

    if (rank == 0) {
        A = new double[N * N] {
            4, -2, 1, -1,
                -2, 4, -2, 1,
                1, -2, 4, -2,
                -1, 1, -2, 4
            };

        b = new double[N] {11, -16, 17, -5};
    }

    // Распределение данных
    double* localA = new double[blockSize * N];
    double* localB = new double[blockSize];

    MPI_Bcast(A, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(b, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(A, blockSize * N, MPI_DOUBLE, localA, blockSize * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(b, blockSize, MPI_DOUBLE, localB, blockSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Решение локальной системы
    for (int k = 0; k < N; ++k) {
        for (int i = 0; i < blockSize; ++i) {
            double scale = localA[i * N + k] / A[k * N + k];
            for (int j = k + 1; j < N; ++j) {
                localA[i * N + j] -= scale * A[k * N + j];
            }
            localB[i] -= scale * b[k];
        }
    }

    // Объединение результатов
    MPI_Gather(localB, blockSize, MPI_DOUBLE, x, blockSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Вывод результата на 0-м процессе
    if (rank == 0) {
        std::cout << "Solution: ";
        printVector(x, N);
    }

    delete[] localA;
    delete[] localB;
    delete[] A;
    delete[] b;
    delete[] x;

    MPI_Finalize();
    return 0;
}
