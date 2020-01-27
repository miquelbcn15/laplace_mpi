#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

/*
 * To compile
    module add openmpi/3.0.0
    mpicc -o lap lapFusion.c -lmpi -lm
 * Execution:
    mpirun -np 4 lap
 */

float stencil ( float v1, float v2, float v3, float v4)
{
  return (v1 + v2 + v3 + v4) * 0.25f;
}

float max_error ( float prev_error, float old, float new )
{
  float t= fabsf( new - old );
  return t>prev_error? t: prev_error;
}

float laplace_step_inner(float *in, float *out, int n/*col*/, 
                   int me, int nprocs)
{
  int i, j;
  float error=0.0f;
  for ( j = 2; j < n/nprocs; j++ )   /* rows */
    for ( i = 1; i < n - 1; i++ ) /* cols */
    {
      out[j*n+i]= stencil(in[j*n+i+1], in[j*n+i-1], in[(j-1)*n+i], in[(j+1)*n+i]);
      error = max_error( error, out[j*n+i], in[j*n+i] );
    }
  return error;
}

float laplace_step_border(float *in, float *out, int n/*col*/, 
                   int me, int nprocs)
{
  int i, j;
  float error = 0.0f;
  if (!me) {
    j = 1;
    for ( i = 1; i < n - 1; i++ ) /* cols */
    {
      out[j*n+i]= stencil(in[j*n+i+1], in[j*n+i-1], in[(j-1)*n+i], in[(j+1)*n+i]);
      error = max_error( error, out[j*n+i], in[j*n+i] );
    }
  }
  if (me != n/nprocs - 1) {
    j = n/nprocs;
    for ( i = 1; i < n - 1; i++ ) /* cols */
    {
      out[j*n+i]= stencil(in[j*n+i+1], in[j*n+i-1], in[(j-1)*n+i], in[(j+1)*n+i]);
      error = max_error( error, out[j*n+i], in[j*n+i] );
    }
  }
  return error;
}

void laplace_init(float *in, int n, int me, int nprocs)
{
  int i;
  const float pi  = 2.0f * asinf(1.0f);
  memset(in, 0, (n/nprocs + 2) * n * sizeof(float));
  for (i=0; i < n/nprocs; i++) {
#define R(j) ((j)*n/nprocs)     // j means actual process number
    float V = in[(i+1)*n + 0/*col*/] = sinf(pi*(i + R(me)) / (n-1));
    in[ (i+1)*n+n-1 ] = V*expf(-pi);
  }
}

int main(int argc, char** argv)
{
  int n = 4000;
  int iter_max = 1000;
  float *A, *temp;
    
  MPI_Status status;
  MPI_Request request0, request1, request2, request3;

  const float tol = 1.0e-5f;
  float error = 1.0f, error1 = 1.0f;    

  // get runtime arguments 
  if (argc>1) {  n        = atoi(argv[1]); }
  if (argc>2) {  iter_max = atoi(argv[2]); }
  
  /* Initialize MPI */
  int me, nprocs;
  MPI_Init(&argc, &argv);
  double t1 = MPI_Wtime();
  MPI_Comm_rank( MPI_COMM_WORLD, &me);
  MPI_Comm_size( MPI_COMM_WORLD, &nprocs);

  if (n % nprocs != 0) {
      if (!me) fprintf(stderr, "ERROR: matriz size cannot be divided by nprocs\n");
      MPI_Finalize();
      return -1;
  } 
  /* Assuming that n can be divided by nprocs, then ... */
#define R0 (n)
#define RF (n*n/nprocs)
  /* Allocating memory */
  A    = (float*) malloc( (n/nprocs + 2) * n * sizeof(float) );
  temp = (float*) malloc( (n/nprocs + 2) * n * sizeof(float) );
  
  /* Set boundary conditions */
  if (!me) fprintf(stderr, "lap(): setting initial conditions\n");
  laplace_init (A, n, me, nprocs);
  laplace_init (temp, n, me, nprocs);
  // if (me == (int)(n/128 / n/nprocs)) A[(n/128 % R(me)) * n + n/128] = 1.0f; 
  if (!me) fprintf(stderr,"Jacobi relaxation Calculation: %d x %d mesh,"
         " maximum of %d iterations\n", 
         n, n, iter_max );

  int iter = 0;
  float gerror = 1.0f;
  fprintf(stderr, "lap(): process %d , init bucle\n", me);
  while ( gerror > tol*tol && iter < iter_max )
  {
    if (me > 0) {
        MPI_Isend(A + R0, n, MPI_FLOAT, me - 1, 33, MPI_COMM_WORLD, &request0);
        MPI_Irecv(A, n, MPI_FLOAT, me - 1, 33, MPI_COMM_WORLD, &request1);
    }
    if (me < nprocs - 1) {
        MPI_Isend(A + RF, n, MPI_FLOAT, me + 1, 33, MPI_COMM_WORLD, &request2);
        MPI_Irecv(A + RF + n, n, MPI_FLOAT, me + 1, 33, MPI_COMM_WORLD, &request3);
    }
    iter++;
    error = laplace_step_inner (A, temp, n /*col*/, me, nprocs);
    if ( me ) MPI_Wait(&request1, &status);
    if ( me != nprocs - 1 ) MPI_Wait(&request3, &status);
    error1 = laplace_step_border (A, temp, n /*col*/, me, nprocs);
    error = error > error1 ? error : error1;
    MPI_Allreduce(&error, &gerror, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
    float *swap = A; A = temp; temp = swap; // swap pointers A & temp
  }
  gerror = sqrtf( gerror );
  if (me == 0)  {
      fprintf(stderr, "# Total Iterations: %5d, ERROR: %0.6f, \n", iter, gerror);
      double t2 = MPI_Wtime();
      fprintf(stdout, "%d %d %lf\n", n, nprocs, t2 - t1);
  }

  free(A); free(temp);
  MPI_Finalize();
}
