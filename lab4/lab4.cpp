#include <cstdio>
#include "mpi.h"

int main(int *argc, char **argv)
{
    int numtasks, rank;

    MPI_Init(argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_rank(MPI_COMM_WORLD, &numtasks);

    printf("rank %d, %d", rank, numtasks);

    MPI_Finalize();

    return 0;
}