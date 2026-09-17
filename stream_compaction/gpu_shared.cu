#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "gpu_shared.h"

namespace StreamCompaction
{
    namespace Shared
    {
#define blockSize 128
#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)

        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer &timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernPrescanShared(int n, int *g_odata, const int *g_idata, int *g_sums)
        {
            extern __shared__ int s_mem[];

            int thid = threadIdx.x;

            int ai = thid;
            int bi = thid + blockDim.x;

            int g_ai = ai + 2 * blockIdx.x * blockDim.x;
            int g_bi = bi + 2 * blockIdx.x * blockDim.x;

            int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
            int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

            s_mem[ai + bankOffsetA] = (g_ai < n) ? g_idata[g_ai] : 0;
            s_mem[bi + bankOffsetB] = (g_bi < n) ? g_idata[g_bi] : 0;

            // Up-sweep
            int offset = 1;
            for (int d = blockDim.x; d > 0; d >>= 1)
            {
                __syncthreads();
                if (thid < d)
                {
                    int a = offset * (2 * thid + 1) - 1;
                    int b = offset * (2 * thid + 2) - 1;
                    a += CONFLICT_FREE_OFFSET(a);
                    b += CONFLICT_FREE_OFFSET(b);

                    s_mem[b] += s_mem[a];
                }
                offset <<= 1;
            }

            // Store total block sum and clear root element
            if (thid == 0)
            {
                int lastIdx = 2 * blockDim.x - 1;
                int totalBankOffset = CONFLICT_FREE_OFFSET(lastIdx);
                if (g_sums != nullptr)
                {
                    g_sums[blockIdx.x] = s_mem[lastIdx + totalBankOffset];
                }
                s_mem[lastIdx + totalBankOffset] = 0;
            }

            // Down-sweep
            for (int d = 1; d <= blockDim.x; d <<= 1)
            {
                offset >>= 1;
                __syncthreads();
                if (thid < d)
                {
                    int a = offset * (2 * thid + 1) - 1;
                    int b = offset * (2 * thid + 2) - 1;
                    a += CONFLICT_FREE_OFFSET(a);
                    b += CONFLICT_FREE_OFFSET(b);

                    int t = s_mem[a];
                    s_mem[a] = s_mem[b];
                    s_mem[b] += t;
                }
            }
            __syncthreads();

            if (g_ai < n)
                g_odata[g_ai] = s_mem[ai + bankOffsetA];
            if (g_bi < n)
                g_odata[g_bi] = s_mem[bi + bankOffsetB];
        }

        __global__ void kernAddBlockOffsets(int n, int *g_odata, const int *g_sums)
        {
            int thid = threadIdx.x;
            int blockOffset = g_sums[blockIdx.x];

            int g_ai = thid + 2 * blockIdx.x * blockDim.x;
            int g_bi = thid + blockDim.x + 2 * blockIdx.x * blockDim.x;

            if (g_ai < n)
                g_odata[g_ai] += blockOffset;
            if (g_bi < n)
                g_odata[g_bi] += blockOffset;
        }

        void scanInternal(int n, int *dev_odata, const int *dev_idata)
        {
            if (n <= 0)
                return;

            int elementsPerBlock = 2 * blockSize;
            int gridSize = (n + elementsPerBlock - 1) / elementsPerBlock;
            int sharedMemBytes = (elementsPerBlock + CONFLICT_FREE_OFFSET(elementsPerBlock - 1)) * sizeof(int);

            if (gridSize == 1)
            {
                kernPrescanShared<<<1, blockSize, sharedMemBytes>>>(n, dev_odata, dev_idata, nullptr);
            }
            else
            {
                int *dev_blockSums = nullptr;
                int *dev_scannedBlockSums = nullptr;

                cudaMalloc((void **)&dev_blockSums, gridSize * sizeof(int));
                cudaMalloc((void **)&dev_scannedBlockSums, gridSize * sizeof(int));

                kernPrescanShared<<<gridSize, blockSize, sharedMemBytes>>>(n, dev_odata, dev_idata, dev_blockSums);

                scanInternal(gridSize, dev_scannedBlockSums, dev_blockSums);

                kernAddBlockOffsets<<<gridSize, blockSize>>>(n, dev_odata, dev_scannedBlockSums);

                cudaFree(dev_blockSums);
                cudaFree(dev_scannedBlockSums);
            }
        }

        void scan(int n, int *odata, const int *idata)
        {
            if (n <= 0)
                return;

            int levels = ilog2ceil(n);
            int paddedN = 1 << levels;

            int *dev_idata = nullptr;
            int *dev_odata = nullptr;

            cudaMalloc((void **)&dev_idata, paddedN * sizeof(int));
            cudaMalloc((void **)&dev_odata, paddedN * sizeof(int));

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            if (paddedN > n)
            {
                cudaMemset(dev_idata + n, 0, (paddedN - n) * sizeof(int));
            }

            timer().startGpuTimer();
            scanInternal(paddedN, dev_odata, dev_idata);
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_idata);
            cudaFree(dev_odata);
        }
    }
}