#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include "radix.h"

namespace StreamCompaction
{
    namespace Radix
    {
#define blockSize 128

        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer &timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernExtractBitAndInvert(int n, int bit, int *e, int *f, const int *idata)
        {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index < n)
            {
                int bitValue = (idata[index] >> bit) & 1;
                e[index] = bitValue;
                f[index] = 1 - bitValue;
            }
        }

        __global__ void kernScatterRadix(int n, int totalZeros, int *odata, const int *idata,
                                         const int *e, const int *f, const int *f_scanned)
        {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index < n)
            {
                int destIndex = (f[index] == 1)
                                    ? f_scanned[index]
                                    : (index - f_scanned[index] + totalZeros);
                odata[destIndex] = idata[index];
            }
        }

        void sort(int n, int *odata, const int *idata)
        {
            if (n <= 0)
                return;

            int levels = ilog2ceil(n);
            int paddedN = 1 << levels;

            int *dev_in = nullptr;
            int *dev_out = nullptr;
            int *dev_e = nullptr;
            int *dev_f = nullptr;
            int *dev_f_scanned = nullptr;

            cudaMalloc((void **)&dev_in, paddedN * sizeof(int));
            cudaMalloc((void **)&dev_out, paddedN * sizeof(int));
            cudaMalloc((void **)&dev_e, paddedN * sizeof(int));
            cudaMalloc((void **)&dev_f, paddedN * sizeof(int));
            cudaMalloc((void **)&dev_f_scanned, paddedN * sizeof(int));

            cudaMemcpy(dev_in, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            if (paddedN > n)
            {
                cudaMemset(dev_in + n, 0, (paddedN - n) * sizeof(int));
            }

            dim3 fullBlocks((n + blockSize - 1) / blockSize);

            timer().startGpuTimer();

            for (int bit = 0; bit < 32; ++bit)
            {
                kernExtractBitAndInvert<<<fullBlocks, blockSize>>>(n, bit, dev_e, dev_f, dev_in);

                // Copy f to f_scanned and pad remainder with zeros
                cudaMemcpy(dev_f_scanned, dev_f, n * sizeof(int), cudaMemcpyDeviceToDevice);
                if (paddedN > n)
                {
                    cudaMemset(dev_f_scanned + n, 0, (paddedN - n) * sizeof(int));
                }

                // Exclusive scan on bit-flag array
                Efficient::runWorkEfficientScan(paddedN, dev_f_scanned);

                // Compute totalZeros = f_scanned[n-1] + f[n-1]
                int lastF = 0, lastFScanned = 0;
                cudaMemcpy(&lastF, dev_f + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(&lastFScanned, dev_f_scanned + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
                int totalZeros = lastF + lastFScanned;

                // Scatter elements to sorted order for bit level
                kernScatterRadix<<<fullBlocks, blockSize>>>(n, totalZeros, dev_out, dev_in, dev_e, dev_f, dev_f_scanned);

                // Ping-pong buffer pointers
                std::swap(dev_in, dev_out);
            }

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_in, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_in);
            cudaFree(dev_out);
            cudaFree(dev_e);
            cudaFree(dev_f);
            cudaFree(dev_f_scanned);
        }
    }
}