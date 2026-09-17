#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define blockSize 128

namespace StreamCompaction
{
    namespace Efficient
    {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer &timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * Up-sweep (Reduction) pass kernel.
         */
        __global__ void kernUpSweep(int numThreads, int d, int *data)
        {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= numThreads)
                return;

            int stride = 1 << (d + 1);
            int offset = 1 << d;
            int k = (index + 1) * stride - 1;
            data[k] += data[k - offset];
        }

        /**
         * Down-sweep pass kernel.
         */
        __global__ void kernDownSweep(int numThreads, int d, int *data)
        {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= numThreads)
                return;

            int stride = 1 << (d + 1);
            int offset = 1 << d;
            int right = (index + 1) * stride - 1;
            int left = right - offset;

            int t = data[left];
            data[left] = data[right];
            data[right] += t;
        }

        /**
         * Device-side helper to run work-efficient exclusive scan on a padded array.
         */
        void runWorkEfficientScan(int paddedN, int *dev_data)
        {
            int max_d = ilog2ceil(paddedN);

            // Up-sweep phase
            for (int d = 0; d < max_d; ++d)
            {
                int numThreads = paddedN >> (d + 1);
                dim3 fullBlocks((numThreads + blockSize - 1) / blockSize);
                kernUpSweep<<<fullBlocks, blockSize>>>(numThreads, d, dev_data);
                checkCUDAError("kernUpSweep failed!");
            }

            // Set root to zero
            cudaMemset(&dev_data[paddedN - 1], 0, sizeof(int));
            checkCUDAError("cudaMemset root zero failed!");

            // Down-sweep phase
            for (int d = max_d - 1; d >= 0; --d)
            {
                int numThreads = paddedN >> (d + 1);
                dim3 fullBlocks((numThreads + blockSize - 1) / blockSize);
                kernDownSweep<<<fullBlocks, blockSize>>>(numThreads, d, dev_data);
                checkCUDAError("kernDownSweep failed!");
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata)
        {
            int levels = ilog2ceil(n);
            int paddedN = 1 << levels;

            int *dev_data = nullptr;
            cudaMalloc((void **)&dev_data, paddedN * sizeof(int));
            checkCUDAError("cudaMalloc dev_data failed!");

            cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy idata to dev_data failed!");

            if (paddedN > n)
            {
                cudaMemset(dev_data + n, 0, (paddedN - n) * sizeof(int));
                checkCUDAError("cudaMemset padding failed!");
            }

            timer().startGpuTimer();
            // TODO
            runWorkEfficientScan(paddedN, dev_data);
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy dev_data to odata failed!");

            cudaFree(dev_data);
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata)
        {
            int levels = ilog2ceil(n);
            int paddedN = 1 << levels;

            int *dev_idata = nullptr;
            int *dev_odata = nullptr;
            int *dev_bools = nullptr;
            int *dev_indices = nullptr;

            cudaMalloc((void **)&dev_idata, n * sizeof(int));
            cudaMalloc((void **)&dev_odata, n * sizeof(int));
            cudaMalloc((void **)&dev_bools, paddedN * sizeof(int));
            cudaMalloc((void **)&dev_indices, paddedN * sizeof(int));
            checkCUDAError("cudaMalloc compaction buffers failed!");

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy idata failed!");

            dim3 fullBlocksN((n + blockSize - 1) / blockSize);

            timer().startGpuTimer();
            // TODO
            // Map to boolean array
            Common::kernMapToBoolean<<<fullBlocksN, blockSize>>>(n, dev_bools, dev_idata);
            checkCUDAError("kernMapToBoolean failed!");

            // Copy mapped booleans to indices buffer and pad extra elements with 0
            cudaMemcpy(dev_indices, dev_bools, n * sizeof(int), cudaMemcpyDeviceToDevice);
            if (paddedN > n)
            {
                cudaMemset(dev_indices + n, 0, (paddedN - n) * sizeof(int));
            }

            // Scan boolean flags in-place
            runWorkEfficientScan(paddedN, dev_indices);

            // Scatter non-zero elements
            Common::kernScatter<<<fullBlocksN, blockSize>>>(n, dev_odata, dev_idata, dev_bools, dev_indices);
            checkCUDAError("kernScatter failed!");

            timer().endGpuTimer();

            // Determine total non-zero elements count: scanned[n-1] + bools[n-1]
            int lastBool = 0, lastIndex = 0;
            cudaMemcpy(&lastBool, dev_bools + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&lastIndex, dev_indices + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            int count = lastBool + lastIndex;

            cudaMemcpy(odata, dev_odata, count * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy dev_odata failed!");

            cudaFree(dev_idata);
            cudaFree(dev_odata);
            cudaFree(dev_bools);
            cudaFree(dev_indices);

            return count;
        }
    }
}
