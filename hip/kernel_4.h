#include "sgemm.hpp"

/* NOTE: The dimBlock.x would be 1/16 of actual data
 * m, n, k 分别表示 shared_memory 中的 A, B 小块的维度 */
__global__ void gemm_kernel4(float *d_A, float *d_B, float *d_C, int M, int N, int K, int m, int n, int k)
{
    int padding = 4;
    // 16 * 16 = 256
    // m * k = 1024 一个线程读 4 个
    // m = 64;
    // n = 64;
    // k = 16;
    const int REG_SIZE = 4;
    const int BATCH = 4; // 每个线程处理四个数据

    extern __shared__ float sh[];
    float *A_sh = sh;
    float *B_sh = sh + m * (k + padding); /* 这里加了 padding */

    int N_tile_index = blockIdx.x; // tile的列号
    int M_tile_index = blockIdx.y; // tile的行号

    int A_m_index;
    int A_n_index;
    int B_m_index;
    int B_n_index;
    int C_m_index = threadIdx.x / ((n + REG_SIZE - 1) / REG_SIZE); // tile内的4 * 4行号
    int C_n_index = threadIdx.x % ((n + REG_SIZE - 1) / REG_SIZE); // tile内的4 * 4列号

    int ix;
    // printf("m_index: %d, n_index: %d\n", m_index, n_index);
    // float reg_A[reg_size][reg_size];
    // float reg_B[reg_size][reg_size];
    float reg_C[REG_SIZE][REG_SIZE] = {0.0f}; /* 在共享内存里面存放 C 的内容 */
    // float total = 0.0f;

    for (int K_tile_index = 0; K_tile_index < int((K + k - 1) / k); K_tile_index++)
    {
        ix = threadIdx.x * BATCH;

        /* 因为 ix 是除过 16 的，现在乘以 4 再分别对 k 进行整除与取模。可以得到
         * 本身呈一维的线程映射到 A_sh 里面的纵横坐标。 */

        A_m_index = ix / k;
        A_n_index = ix % k;

        /* 计算 Global Memory 中 A 对应的坐标，这里只使用了一个坐标，因为输入的时候只有一个坐标。 */
        int d_A_index = (M_tile_index * m + A_m_index) * K /* 横坐标乘以K */
                        + K_tile_index * k + A_n_index;    /* 加纵坐标 */

        ix = A_m_index * (k + padding) + A_n_index; /* A_sh 中，对应坐标 [A_m_index, A_n_index] 位置 */

        /* NOTE: 上面这里为何要加 padding ？因为下面调用这个核的时候，就给 shared_memory 加了 padding. 当然，如果不加好
         * 像也可以，因为 shared_memory 其实就是一个暂存的位置罢了，只要最后映射结果正确就好了。 */

        /* 将四个数据从 Global Memory 载入 Shared Memory 里面，使用我们刚刚计算得到的坐标 */
        FLOAT4(A_sh[ix]) = FLOAT4(d_A[d_A_index]);

        /* 这里要加这么一句，如果不加就是错的吧？下面都是如法炮制。 */
        ix = threadIdx.x * BATCH;

        B_m_index = ix / n;
        B_n_index = ix % n;

        ix = B_m_index * (n + padding) + B_n_index;
        int d_B_index = (K_tile_index * k + B_m_index) * N + N_tile_index * n + B_n_index;

        FLOAT4(B_sh[ix]) = FLOAT4(d_B[d_B_index]);

        /* 算法的核心开始，核心的思想就是对这一个小块的 A_sh 与 B_sh 做矩阵乘法，并累积到 C 中。 */
        __syncthreads();
        for (int k_reg_index = 0; k_reg_index < k; k_reg_index += REG_SIZE)
        {
            for (int i = 0; i < REG_SIZE; i++)
            {
                for (int j = 0; j < REG_SIZE; j++)
                {
                    for (int k_index = 0; k_index < REG_SIZE; k_index++)
                    {
                        int A_index = C_m_index * REG_SIZE * (k + padding) + k_reg_index + i * (k + padding) + k_index;
                        int B_index = k_reg_index * (n + padding) + C_n_index * REG_SIZE + k_index * (n + padding) + j;

                        reg_C[i][j] += A_sh[A_index] * B_sh[B_index];
                    }
                }
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < REG_SIZE; i++)
    {
        int C_index = (M_tile_index * m + C_m_index * REG_SIZE) * N + N_tile_index * n + C_n_index * REG_SIZE + i * N;
        FLOAT4(d_C[C_index]) = FLOAT4(reg_C[i][0]);
    }
}
float test4()
{
    const int m = 64;
    const int n = 64;
    const int k = 16;
    const int reg_size = 4;
    const int padding = 4;
    // int thread_size = (m * n + reg_size * reg_size - 1) / (reg_size * reg_size);
    test_start();
    int block_size = min(m * n, C_size);

    dim3 dimGrid((M + m - 1) / m, (N + n - 1) / n);
    /* 每个线程处理 reg_size * reg_size 大小的数据，此处结果必为 16 */
    dim3 dimBlock((block_size + reg_size * reg_size - 1) / (reg_size * reg_size));
    int shared_size = sizeof(float) * (m * (k + padding) + k * (n + padding));

    gemm_kernel4<<<dimGrid, dimBlock, shared_size>>>(d_A, d_B, d_C, M, N, K, m, n, k);

    KernelErrChk();
    ErrChk(hipEventRecord(start, 0));
    for (int i = 0; i < iteration; i++)
    {
        gemm_kernel4<<<dimGrid, dimBlock, shared_size>>>(d_A, d_B, d_C, M, N, K, m, n, k);
    }
    test_end();
    return elapsedTime / iteration;
}
