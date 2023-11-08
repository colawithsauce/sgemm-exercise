#ifndef __SGEMM_HPP__
#define __SGEMM_HPP__

#include <hip/hip_runtime.h>
#include <omp.h>
#include <rocblas.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <rocwmma/rocwmma.hpp>

using namespace std;
#define iteration 1000
const int reg_size = 2;
const int M = 1 << 6;
const int K = 1 << 6;
const int N = 1 << 6;
const int m = 16;
const int n = 16;
const int k = 16;
const int WAVE_SIZE = rocwmma::Constants::AMDGCN_WAVE_SIZE;
typedef float Float4 __attribute__((ext_vector_type(4)));
typedef float Float2 __attribute__((ext_vector_type(2)));
// #define __local __attribute__((address_space(3)))
// __device__ inline static __local void* __to_local(unsigned x) { return
// (__local void*)x; }
__device__ inline static __local void *__to_local(float *x)
{
    return (__local void *)x;
}
#define OFFSET(row, col, ld) ((row) * (ld) + (col))

// transfer float4
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])

#define FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])
#define FLOAT2(pointer) (reinterpret_cast<float2 *>(&(pointer))[0])
#define ErrChk(code)                                                                                                   \
    {                                                                                                                  \
        Assert((code), __FILE__, __LINE__);                                                                            \
    }
static inline void Assert(hipError_t code, const char *file, int line)
{
    if (code != hipSuccess)
    {
        printf("CUDA Runtime Error: %s:%d:'%s'\n", file, line, hipGetErrorString(code));
        exit(EXIT_FAILURE);
    }
}
// static inline void Assert(miopenStatus_t code, const char *file, int line){
//     if (code!=miopenStatusSuccess){
// 		printf("cuDNN API Error: %s:%d:'%s'\n", file, line,
// miopenGetErrorString(code));
//         exit(EXIT_FAILURE);
//     }
// }
// static inline void Assert(hipblasStatus_t code, const char *file, int line){
//     if (code!=HIPBLAS_STATUS_SUCCESS){
// 		printf("cuBLAS API Error: %s:%d:'%s'\n", file, line,
// hipblasGetErrorString(code));
//         exit(EXIT_FAILURE);
//     }
// }

#define KernelErrChk()                                                                                                 \
    {                                                                                                                  \
        hipError_t errSync = hipGetLastError();                                                                        \
        hipError_t errAsync = hipDeviceSynchronize();                                                                  \
        if (errSync != hipSuccess)                                                                                     \
        {                                                                                                              \
            printf("Sync kernel error: %s\n", hipGetErrorString(errSync));                                             \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
        if (errAsync != hipSuccess)                                                                                    \
        {                                                                                                              \
            printf("Async kernel error: %s\n", hipGetErrorString(errAsync));                                           \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    }

#define test_start()                                                                                                   \
    float *h_A;                                                                                                        \
    float *h_B;                                                                                                        \
    float *h_C;                                                                                                        \
    float *test_C;                                                                                                     \
    int A_size = M * K;                                                                                                \
    int B_size = K * N;                                                                                                \
    int C_size = M * N;                                                                                                \
    int A_bytes = sizeof(float) * A_size;                                                                              \
    int B_bytes = sizeof(float) * B_size;                                                                              \
    int C_bytes = sizeof(float) * C_size;                                                                              \
    h_A = (float *)malloc(A_bytes);                                                                                    \
    h_B = (float *)malloc(B_bytes);                                                                                    \
    h_C = (float *)malloc(C_bytes);                                                                                    \
    test_C = (float *)malloc(C_bytes);                                                                                 \
    for (int i = 0; i < A_size; i++)                                                                                   \
    {                                                                                                                  \
        h_A[i] = rand() % 3 * 0.1;                                                                                     \
    }                                                                                                                  \
    for (int i = 0; i < B_size; i++)                                                                                   \
    {                                                                                                                  \
        h_B[i] = rand() % 4 * 0.01;                                                                                    \
    }                                                                                                                  \
    float *d_A;                                                                                                        \
    float *d_B;                                                                                                        \
    float *d_C;                                                                                                        \
    ErrChk(hipMalloc(&d_A, A_bytes));                                                                                  \
    ErrChk(hipMalloc(&d_B, B_bytes));                                                                                  \
    ErrChk(hipMalloc(&d_C, C_bytes));                                                                                  \
    ErrChk(hipMemcpy(d_A, h_A, A_bytes, hipMemcpyHostToDevice));                                                       \
    ErrChk(hipMemcpy(d_B, h_B, B_bytes, hipMemcpyHostToDevice));                                                       \
    hipEvent_t start, stop;                                                                                            \
    float elapsedTime = 0.0f;                                                                                          \
    ErrChk(hipEventCreate(&start));                                                                                    \
    ErrChk(hipEventCreate(&stop));

#define test_end()                                                                                                     \
    ErrChk(hipEventRecord(stop, 0));                                                                                   \
    ErrChk(hipEventSynchronize(stop));                                                                                 \
    ErrChk(hipEventElapsedTime(&elapsedTime, start, stop));                                                            \
    ErrChk(hipMemcpy(h_C, d_C, C_bytes, hipMemcpyDeviceToHost));                                                       \
    /*for (int i = 0; i < M; i++) {                                                                                    \
        for (int j = 0; j < N; j++) {                                                                                  \
            float total = 0;                                                                                           \
            for (int k = 0; k < K; k++) {                                                                              \
                total += h_A[i * K + k] * h_B[k * N + j];                                                              \
            }                                                                                                          \
            test_C[i * N + j] = total;                                                                                 \
        }                                                                                                              \
    }                                                                                                                  \
    bool isSame = true;                                                                                                \
    for (int i = 0; i < C_size; i++) {                                                                                 \
        if (abs(test_C[i] - h_C[i]) > 0.01) {                                                                          \
            cout << "error: i: " << i << " test_C: " << test_C[i] << " h_C[i]: "                                       \
    << h_C[i] << endl; isSame = false; break;                                                                          \
        }                                                                                                              \
    }*/                                                                                                                \
    ErrChk(hipFree(d_A));                                                                                              \
    ErrChk(hipFree(d_B));                                                                                              \
    ErrChk(hipFree(d_C));                                                                                              \
    ErrChk(hipEventDestroy(start));                                                                                    \
    ErrChk(hipEventDestroy(stop));                                                                                     \
    free(h_A);                                                                                                         \
    free(h_B);                                                                                                         \
    free(h_C);

__device__ void set_value(float *dst, float *source, const int n)
{
    int i = 0;
    if (n == 1)
    {
        dst[0] = source[0];
    }
    else if (n == 2)
    {
        FLOAT2(dst[0]) = FLOAT2(source[0]);
    }
    else if (n == 4)
    {
        FLOAT4(dst[0]) = FLOAT4(source[0]);
    }
    else
    {
        while (i < n)
        {
            if (i + 3 < n)
            {
                FLOAT4(dst[i]) = FLOAT4(source[i]);
                i += 4;
            }
            else if (i + 1 < n)
            {
                FLOAT2(dst[i]) = FLOAT2(source[i]);
                i += 2;
            }
            else if (i < n)
            {
                dst[i] = source[i];
                i++;
            }
        }
    }
}

__device__ void set_value_matrix(float *dst, float *source, int dst_m, int dst_n, int dst_lda, int source_lda)
{
    for (int i = 0; i < dst_m; i++)
    {
        set_value(&dst[i * dst_lda], &source[i * source_lda], dst_n);
    }
}
template <uint32_t offset> inline __device__ void global_load(float *ptr, float4 &val)
{
    if (offset == 0)
    {
        asm volatile("\n \
    global_load_dwordx4 %0, %1, off \n \
    "
                     : "=v"(val)
                     : "v"(ptr));
        return;
    }
    if (offset == 8)
    {
        asm volatile("\n \
    global_load_dwordx4 %0, %1, off offset:32 \n \
    "
                     : "=v"(val)
                     : "v"(ptr));
    }
}

template <uint32_t offset> inline __device__ void global_load(float *ptr, Float4 &val)
{
    if (offset == 0)
    {
        asm volatile("\n \
    global_load_dwordx4 %0, %1, off \n \
    "
                     : "=v"(val)
                     : "v"(ptr));
        return;
    }
    if (offset == 8)
    {
        asm volatile("\n \
    global_load_dwordx4 %0, %1, off offset:32 \n \
    "
                     : "=v"(val)
                     : "v"(ptr));
    }
}
template <uint32_t offset> inline __device__ void global_store(float *ptr, float4 val)
{
    Float4 mid;
    mid.x = val.x;
    mid.y = val.y;
    mid.z = val.z;
    mid.w = val.w;
    if (offset == 0 * 32)
    {
        asm volatile("\n \
    global_store_dwordx4 %1, %0, off \n \
    "
                     :
                     : "v"(mid), "v"(ptr));
        return;
    }
    if (offset == 16)
    {
        asm volatile("\n \
    global_store_dwordx4 %1, %0, off offset:16*4*4 \n \
    "
                     :
                     : "v"(mid), "v"(ptr));
    }
}

template <uint32_t cnt> inline __device__ void lgkmcnt()
{
    if (cnt == 0)
    {
        asm volatile("\n \
    s_waitcnt lgkmcnt(0) \n \
    " ::);
    }
    if (cnt == 1)
    {
        asm volatile("\n \
    s_waitcnt lgkmcnt(1) \n \
    " ::);
    }
    if (cnt == 2)
    {
        asm volatile("\n \
    s_waitcnt lgkmcnt(2) \n \
    " ::);
    }
    if (cnt == 3)
    {
        asm volatile("\n \
    s_waitcnt lgkmcnt(3) \n \
    " ::);
    }
    if (cnt == 4)
    {
        asm volatile("\n \
    s_waitcnt lgkmcnt(4) \n \
    " ::);
    }
    if (cnt == 5)
    {
        asm volatile("\n \
    s_waitcnt lgkmcnt(5) \n \
    " ::);
    }
    if (cnt == 6)
    {
        asm volatile("\n \
    s_waitcnt lgkmcnt(6) \n \
    " ::);
    }

    /**
    * Disabling as 16 is to high to fit in 4bits (15 max)
      if(cnt == 16) {
        asm volatile("\n \
        s_waitcnt lgkmcnt(16) \n \
        "::);
      }
    */
}

template <uint32_t cnt> inline __device__ void vmcnt()
{
    if (cnt == 0)
    {
        asm volatile("\n \
      s_waitcnt vmcnt(0) \n \
      " ::);
    }
    if (cnt == 1)
    {
        asm volatile("\n \
      s_waitcnt vmcnt(1) \n \
      " ::);
    }
    if (cnt == 2)
    {
        asm volatile("\n \
      s_waitcnt vmcnt(2) \n \
      " ::);
    }
    if (cnt == 4)
    {
        asm volatile("\n \
      s_waitcnt vmcnt(2) \n \
      " ::);
    }
}
template <uint32_t cnt> inline __device__ void vscnt()
{
    if (cnt == 0)
    {
        asm volatile("\n \
      s_waitcnt vscnt(0) \n \
      " ::);
    }
    if (cnt == 1)
    {
        asm volatile("\n \
      s_waitcnt vscnt(1) \n \
      " ::);
    }
    if (cnt == 2)
    {
        asm volatile("\n \
      s_waitcnt vscnt(2) \n \
      " ::);
    }
    if (cnt == 4)
    {
        asm volatile("\n \
      s_waitcnt vscnt(2) \n \
      " ::);
    }
}
inline __device__ void fma_op(float &c, float &a, float &b)
{
    asm volatile("\n \
          v_fma_f32 %0, %1, %2, %0  \n \
          "
                 :
                 : "v"(c), "v"(a), "v"(b));
}
#endif // !__SGEMM_HPP__
