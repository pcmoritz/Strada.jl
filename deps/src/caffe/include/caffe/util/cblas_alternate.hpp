#if BLASSUFFIX
    #define BLAS(x) x ## 64_
#else
    #define BLAS(x) x
#endif

#if BLASSUFFIX

#define blasint CBLAS_INDEX

double BLAS(cblas_dasum)(const int N, const double *X, const int incX);

void BLAS(cblas_daxpy)(const int N, const double alpha, const double *X,
                 const int incX, double *Y, const int incY);

void BLAS(cblas_dcopy)(const int N, const double *X, const int incX,
                 double *Y, const int incY);

double BLAS(cblas_ddot)(const int N, const double *X, const int incX,
                  const double *Y, const int incY);

void BLAS(cblas_dgemm)(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const double alpha, const double *A,
                 const int lda, const double *B, const int ldb,
                 const double beta, double *C, const int ldc);

void BLAS(cblas_dgemv)(const enum CBLAS_ORDER Order,
                 const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                 const double alpha, const double *A, const int lda,
                 const double *X, const int incX, const double beta,
                 double *Y, const int incY);

void BLAS(cblas_dscal)(const int N, const double alpha, double *X, const int incX);

void BLAS(cblas_daxpby)(const int n, const double a, const double *x, const int incx, const double b, double *y, const int incy);


float BLAS(cblas_sasum)(const int N, const float *X, const int incX);

void BLAS(cblas_saxpy)(const int N, const float alpha, const float *X,
                 const int incX, float *Y, const int incY);

void BLAS(cblas_scopy)(const int N, const float *X, const int incX,
                 float *Y, const int incY);

float BLAS(cblas_sdot)(const int N, const float  *X, const int incX,
                  const float  *Y, const int incY);

void BLAS(cblas_sgemm)(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_TRANSPOSE TransB, const blasint M, const blasint N,
                 const blasint K, const float alpha, const float *A,
                 const blasint lda, const float *B, const blasint ldb,
                 const float beta, float *C, const blasint ldc);

void BLAS(cblas_sgemv)(const enum CBLAS_ORDER Order,
                 const enum CBLAS_TRANSPOSE TransA, const blasint M, const blasint N,
                 const float alpha, const float *A, const blasint lda,
                 const float *X, const blasint incX, const float beta,
                 float *Y, const blasint incY);

void BLAS(cblas_sscal)(const int N, const float alpha, float *X, const int incX);

void BLAS(cblas_saxpby)(const int n, const float a, const float *x, const int incx, const float b, float *y, const int incy);

#endif
