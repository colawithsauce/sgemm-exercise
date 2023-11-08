#include "sgemm.hpp"
#include "kernel_1.h"
#include "kernel_10.h"
#include "kernel_2.h"
#include "kernel_3.h"
#include "kernel_4.h"
#include "kernel_5.h"
#include "kernel_6.h"
#include "kernel_7.h"
#include "kernel_8.h"
#include "kernel_9.h"

float rocblas_result()
{
    test_start();
    int thread_size = 32;
    rocblas_handle handle;
    rocblas_create_handle(&handle);
    float alpha = 1.0f;
    float beta = 0.0f;
    rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_none, M, N, K, &alpha, d_A, M, d_B, K, &beta, d_C,
                  M);
    KernelErrChk();
    hipEventRecord(start, 0);
    for (int i = 0; i < iteration; i++)
    {
        rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_none, M, N, K, &alpha, d_A, M, d_B, K, &beta,
                      d_C, M);
    }
    test_end();
    rocblas_destroy_handle(handle);
    return elapsedTime / iteration;
}


int main()
{
    float baseTime;
    float preTime;
    float nowTime;
    float Tflops;
    // 一些记录:
    // 一个块最多由1024个线程

    cout << "warp: " << WAVE_SIZE << endl;
    // rocblas版本
    baseTime = rocblas_result();
    Tflops = 2 * ((float)M * N * K) / (baseTime / 1000) / 1e12;
    cout << "rocblas: " << baseTime << "ms"
         << ", Tflops: " << Tflops << endl;
    preTime = test1();
    // 1. 无优化版本
    cout << "test1: " << preTime << "ms" << endl;
    // 2. 共享内存分块
    preTime = baseTime;
    nowTime = test2();
    Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    cout << "test2: " << nowTime << "ms speedup: " << preTime / nowTime << ", rocblas_ratio: " << baseTime / nowTime
         << ", Tflops: " << Tflops << ", Tflops_ratio: " << Tflops / 23.1 << endl;
    preTime = nowTime;
    // 3. 线程分块, 一个线程加好几个值
    nowTime = test3();
    Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    cout << "test3: " << nowTime << "ms speedup: " << preTime / nowTime << ", rocblas_ratio: " << baseTime / nowTime
         << ", Tflops: " << Tflops << ", Tflops_ratio: " << Tflops / 23.1 << endl;
    preTime = nowTime;
    // 3.1 寄存器缓存
    nowTime = test3_1();
    Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    cout << "test3_1: " << nowTime << "ms speedup: " << preTime / nowTime << ", rocblas_ratio: " << baseTime / nowTime
         << ", Tflops: " << Tflops << ", Tflops_ratio: " << Tflops / 23.1 << endl;
    preTime = nowTime;
    // 3.2 128 * 8 * 128
    nowTime = test3_2();
    Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    cout << "test3_2: " << nowTime << "ms speedup: " << preTime / nowTime << ", rocblas_ratio: " << baseTime / nowTime
         << ", Tflops: " << Tflops << ", Tflops_ratio: " << Tflops / 23.1 << endl;
    preTime = nowTime;
    // 3.3 128 * 8 * 128 用寄存器缓存
    nowTime = test3_3();
    Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    cout << "test3_3: " << nowTime << "ms speedup: " << preTime / nowTime << ", rocblas_ratio: " << baseTime / nowTime
         << ", Tflops: " << Tflops << ", Tflops_ratio: " << Tflops / 23.1 << endl;
    preTime = nowTime;
    // 3.4 128 * 8 * 128 padding + A 转置
    nowTime = test3_4();
    Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    cout << "test3_4: " << nowTime << "ms speedup: " << preTime / nowTime << ", rocblas_ratio: " << baseTime / nowTime
         << ", Tflops: " << Tflops << ", Tflops_ratio: " << Tflops / 23.1 << endl;
    preTime = nowTime;
    // 3.5 128 * 8 * 128 warp分块
    nowTime = test3_5();
    Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    cout << "test3_5: " << nowTime << "ms speedup: " << preTime / nowTime << ", rocblas_ratio: " << baseTime / nowTime
         << ", Tflops: " << Tflops << ", Tflops_ratio: " << Tflops / 23.1 << endl;
    preTime = nowTime;
    // 3.6 128 * 8 * 128
    nowTime = test3_6();
    Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    cout << "test3_6: " << nowTime << "ms speedup: " << preTime / nowTime << ", rocblas_ratio: " << baseTime / nowTime
         << ", Tflops: " << Tflops << ", Tflops_ratio: " << Tflops / 23.1 << endl;
    preTime = nowTime;

    // 4. 共享内存冲突处理 padding:4
    nowTime = test4();
    Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    cout << "test4: " << nowTime << "ms speedup: " << preTime / nowTime << ", rocblas_ratio: " << baseTime / nowTime
         << ", Tflops: " << Tflops << ", Tflops_ratio: " << Tflops / 23.1 << endl;
    preTime = nowTime;
    // 5. 寄存器缓存
    nowTime = test5();
    Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    cout << "test5: " << nowTime << "ms speedup: " << preTime / nowTime << ", rocblas_ratio: " << baseTime / nowTime
         << ", Tflops: " << Tflops << ", Tflops_ratio: " << Tflops / 23.1 << endl;
    preTime = nowTime;
    // 6. 数据预取或双缓存: 负优化
    nowTime = test6();
    Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    cout << "test6: " << nowTime << "ms speedup: " << preTime / nowTime << ", rocblas_ratio: " << baseTime / nowTime
         << ", Tflops: " << Tflops << ", Tflops_ratio: " << Tflops / 23.1 << endl;
    preTime = nowTime;
    // 7. 调整寄存器计算时的块
    nowTime = test7();
    Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    cout << "test7: " << nowTime << "ms speedup: " << preTime / nowTime << ", rocblas_ratio: " << baseTime / nowTime
         << ", Tflops: " << Tflops << ", Tflops_ratio: " << Tflops / 23.1 << endl;
    preTime = nowTime;
    // 7.1 调整参数
    nowTime = test7_1();
    Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    cout << "test7_1: " << nowTime << "ms speedup: " << preTime / nowTime << ", rocblas_ratio: " << baseTime / nowTime
         << ", Tflops: " << Tflops << ", Tflops_ratio: " << Tflops / 23.1 << endl;
    preTime = nowTime;
    // 7.2 调整参数
    nowTime = test7_2();
    Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    cout << "test7_2: " << nowTime << "ms speedup: " << preTime / nowTime << ", rocblas_ratio: " << baseTime / nowTime
         << ", Tflops: " << Tflops << ", Tflops_ratio: " << Tflops / 23.1 << endl;
    preTime = nowTime;
    // 7.3
    nowTime = test7_3();
    Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    cout << "test7_3: " << nowTime << "ms speedup: " << preTime / nowTime << ", rocblas_ratio: " << baseTime / nowTime
         << ", Tflops: " << Tflops << ", Tflops_ratio: " << Tflops / 23.1 << endl;
    preTime = nowTime;
    // 7.4
    nowTime = test7_4();
    Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    cout << "test7_4: " << nowTime << "ms speedup: " << preTime / nowTime << ", rocblas_ratio: " << baseTime / nowTime
         << ", Tflops: " << Tflops << ", Tflops_ratio: " << Tflops / 23.1 << endl;
    preTime = nowTime;
    // 7.5
    nowTime = test7_5();
    Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    cout << "test7_5: " << nowTime << "ms speedup: " << preTime / nowTime << ", rocblas_ratio: " << baseTime / nowTime
         << ", Tflops: " << Tflops << ", Tflops_ratio: " << Tflops / 23.1 << endl;
    preTime = nowTime;
    // 7.6
    nowTime = test7_6();
    Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    cout << "test7_6: " << nowTime << "ms speedup: " << preTime / nowTime << ", rocblas_ratio: " << baseTime / nowTime
         << ", Tflops: " << Tflops << ", Tflops_ratio: " << Tflops / 23.1 << endl;
    preTime = nowTime;
    // 7.7
    nowTime = test7_7();
    Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    cout << "test7_7: " << nowTime << "ms speedup: " << preTime / nowTime << ", rocblas_ratio: " << baseTime / nowTime
         << ", Tflops: " << Tflops << ", Tflops_ratio: " << Tflops / 23.1 << endl;
    preTime = nowTime;
    // 7.8
    nowTime = test7_8();
    Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    cout << "test7_8: " << nowTime << "ms speedup: " << preTime / nowTime << ", rocblas_ratio: " << baseTime / nowTime
         << ", Tflops: " << Tflops << ", Tflops_ratio: " << Tflops / 23.1 << endl;
    preTime = nowTime;
    // 8. 分为 warp 块执行, 无效果
    nowTime = test8();
    Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    cout << "test8: " << nowTime << "ms speedup: " << preTime / nowTime << ", rocblas_ratio: " << baseTime / nowTime
         << ", Tflops: " << Tflops << ", Tflops_ratio: " << Tflops / 23.1 << endl;
    preTime = nowTime;
    // 8.1 对齐共享内存地址
    nowTime = test8_1();
    Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    cout << "test8_1: " << nowTime << "ms speedup: " << preTime / nowTime << ", rocblas_ratio: " << baseTime / nowTime
         << ", Tflops: " << Tflops << ", Tflops_ratio: " << Tflops / 23.1 << endl;
    preTime = nowTime;
    // 8.2 使用向量外积
    nowTime = test8_2();
    Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    cout << "test8_2: " << nowTime << "ms speedup: " << preTime / nowTime << ", rocblas_ratio: " << baseTime / nowTime
         << ", Tflops: " << Tflops << ", Tflops_ratio: " << Tflops / 23.1 << endl;
    preTime = nowTime;
    // 8.3 sh_B 转置(负优化)
    nowTime = test8_3();
    Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    cout << "test8_3: " << nowTime << "ms speedup: " << preTime / nowTime << ", rocblas_ratio: " << baseTime / nowTime
         << ", Tflops: " << Tflops << ", Tflops_ratio: " << Tflops / 23.1 << endl;
    preTime = nowTime;
    // 8.4  warp 修改尺寸
    nowTime = test8_4();
    Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    cout << "test8_4: " << nowTime << "ms speedup: " << preTime / nowTime << ", rocblas_ratio: " << baseTime / nowTime
         << ", Tflops: " << Tflops << ", Tflops_ratio: " << Tflops / 23.1 << endl;
    preTime = nowTime;
    // 8.5 warp z字布局(不能说有没有优化)
    nowTime = test8_5();
    Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    cout << "test8_5: " << nowTime << "ms speedup: " << preTime / nowTime << ", rocblas_ratio: " << baseTime / nowTime
         << ", Tflops: " << Tflops << ", Tflops_ratio: " << Tflops / 23.1 << endl;
    preTime = nowTime;
    // 8.6
    nowTime = test8_6();
    Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    cout << "test8_6: " << nowTime << "ms speedup: " << preTime / nowTime << ", rocblas_ratio: " << baseTime / nowTime
         << ", Tflops: " << Tflops << ", Tflops_ratio: " << Tflops / 23.1 << endl;
    preTime = nowTime;

    // 9. 将4 * 4 分开拆成 2 * 2 的块执行
    nowTime = test9();
    Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    cout << "test9: " << nowTime << "ms speedup: " << preTime / nowTime << ", rocblas_ratio: " << baseTime / nowTime
         << ", Tflops: " << Tflops << ", Tflops_ratio: " << Tflops / 23.1 << endl;
    preTime = nowTime;
    // 9.1
    nowTime = test9_1();
    Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    cout << "test9_1: " << nowTime << "ms speedup: " << preTime / nowTime << ", rocblas_ratio: " << baseTime / nowTime
         << ", Tflops: " << Tflops << ", Tflops_ratio: " << Tflops / 23.1 << endl;
    preTime = nowTime;
    // 9.2
    nowTime = test9_2();
    Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    cout << "test9_2: " << nowTime << "ms speedup: " << preTime / nowTime << ", rocblas_ratio: " << baseTime / nowTime
         << ", Tflops: " << Tflops << ", Tflops_ratio: " << Tflops / 23.1 << endl;
    preTime = nowTime;
    // 9.3
    nowTime = test9_3();
    Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    cout << "test9_3: " << nowTime << "ms speedup: " << preTime / nowTime << ", rocblas_ratio: " << baseTime / nowTime
         << ", Tflops: " << Tflops << ", Tflops_ratio: " << Tflops / 23.1 << endl;
    preTime = nowTime;

    // 10
    nowTime = test10();
    Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    cout << "test10: " << nowTime << "ms speedup: " << preTime / nowTime << ", rocblas_ratio: " << baseTime / nowTime
         << ", Tflops: " << Tflops << ", Tflops_ratio: " << Tflops / 23.1 << endl;
    preTime = nowTime;
    // // 10.2
    // nowTime = test10_2();
    // Tflops = 2 * ((float)M * N * K) / (nowTime / 1000) / 1e12;
    // cout << "test10_2: " << nowTime  << "ms speedup: " << preTime / nowTime <<
    // ", Glops: " << Tflops << endl; preTime = nowTime;

    return 0;
}
