#include "sgemm.hpp"

/* 这个 kernel 的正确性是建立在 m == n == ( k >> 2 ) 的基础上的，看似一个线程分配了 16个输出数据位置，但是每轮循环都只处
 * 理了 4 个输入。 */

/* kernel 5: 使用寄存器缓存数据, 之前虽然线程又分块, 但是直接使用共享内存. 现在会先读取到寄存器再完成计算。 */
__global__ void gemm_kernel5(float *d_A, float *d_B, float *d_C, int M, int N, int K, int m, int n, int k)
{
    int padding = 4;
    // 16 * 16 = 256
    // m * k = 1024 一个线程读 4 个
    // m = 64;
    // n = 64;
    // k = 16;
    const int REG_SIZE = 4;
    extern __shared__ float sh[];

    float *A_sh = sh;
    float *B_sh = sh + m * (k + padding);

    int N_tile_index = blockIdx.x; // tile的列号
    int M_tile_index = blockIdx.y; // tile的行号

    int A_m_index;
    int A_n_index;
    int B_m_index;
    int B_n_index;

    int C_m_index = threadIdx.x / ((n + REG_SIZE - 1) / REG_SIZE); // tile内的4 * 4行号
    int C_n_index = threadIdx.x % ((n + REG_SIZE - 1) / REG_SIZE); // tile内的4 * 4列号

    int per_thread_num;
    int ix;

    float reg_A[REG_SIZE][REG_SIZE];
    float reg_B[REG_SIZE][REG_SIZE];
    float reg_C[REG_SIZE][REG_SIZE] = {0.0f};

    /* 将 K 分成一个个维度为 k 的小块 */
    for (int K_tile_index = 0; K_tile_index < int((K + k - 1) / k); K_tile_index++)
    {
        /* 从 Global_memory 向 shared_memory 里面载入 A */
        per_thread_num = (m * k + blockDim.x - 1) / blockDim.x;
        ix = threadIdx.x * per_thread_num;
        A_m_index = ix / k;
        A_n_index = ix % k;
        int d_A_index = (M_tile_index * m + A_m_index) * K + K_tile_index * k + A_n_index;
        ix = A_m_index * (k + padding) + A_n_index;
        FLOAT4(A_sh[ix]) = FLOAT4(d_A[d_A_index]);

        /* 从 Global_memory 向 shared_memory 里面载入 B */
        per_thread_num = (k * n + blockDim.x - 1) / blockDim.x;
        ix = threadIdx.x * per_thread_num;
        B_m_index = ix / n;
        B_n_index = ix % n;
        ix = B_m_index * (n + padding) + B_n_index;
        int d_B_index = (K_tile_index * k + B_m_index) * N + N_tile_index * n + B_n_index;
        FLOAT4(B_sh[ix]) = FLOAT4(d_B[d_B_index]);

        /* 等待所有的数据加载完成 */
        __syncthreads();

        /* kernel 5: 使用寄存器缓存数据, 之前虽然线程又分块, 但是直接使用共享内存。 现在会先读取到寄存器再完成计算。 */

        /* 对 shared_memory 中的矩阵，又用 4*4 的分块来分块计算。现在的遍历，是为了对本线程负责的 4*4 patch of 输出矩阵
         * C，做矩阵乘法运算。 */
        for (int k_reg_index = 0; k_reg_index < k; k_reg_index += REG_SIZE)
        {
            /* 将 shared_memory 中，本轮 4*4 patch 的内容加载入 reg_A, reg_B 里面 */
            for (int i = 0; i < REG_SIZE; i++)
            {
                int A_index = C_m_index * REG_SIZE * (k + padding) + k_reg_index + i * (k + padding);
                int B_index = k_reg_index * (n + padding) + C_n_index * REG_SIZE + i * (n + padding);

                FLOAT4(reg_A[i][0]) = FLOAT4(A_sh[A_index]);
                FLOAT4(reg_B[i][0]) = FLOAT4(B_sh[B_index]);
            }

            /* 使用刚才加载的两个块，做一次矩阵乘法，写入对应的 reg_C 中。 */
            for (int i = 0; i < REG_SIZE; i++)
            {
                for (int j = 0; j < REG_SIZE; j++)
                {
                    reg_C[i][j] += reg_A[i][0] * reg_B[0][j];
                    reg_C[i][j] += reg_A[i][1] * reg_B[1][j];
                    reg_C[i][j] += reg_A[i][2] * reg_B[2][j];
                    reg_C[i][j] += reg_A[i][3] * reg_B[3][j];
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

float test5()
{
    const int m = 64;
    const int n = 64;
    const int k = 16;
    const int REG_SIZE = 4;
    const int padding = 4;
    // int thread_size = (m * n + reg_size * reg_size - 1) / (reg_size * reg_size);

    test_start();

    int n_data = min(m * n, C_size);
    dim3 dimGrid((M + m - 1) / m, (N + n - 1) / n);
    /* 将 16 个 data 作为一个 patch, assign 给一个 thread. */
    dim3 dimBlock((n_data + REG_SIZE * REG_SIZE - 1) / (REG_SIZE * REG_SIZE));
    int shared_size = sizeof(float) * (m * (k + padding) + k * (n + padding));

    gemm_kernel5<<<dimGrid, dimBlock, shared_size>>>(d_A, d_B, d_C, M, N, K, m, n, k);

    KernelErrChk();

    ErrChk(hipEventRecord(start, 0));
    for (int i = 0; i < iteration; i++)
    {
        gemm_kernel5<<<dimGrid, dimBlock, shared_size>>>(d_A, d_B, d_C, M, N, K, m, n, k);
    }
    test_end();
    return elapsedTime / iteration;
}
