#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#define blockSize 128

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
        /**
         * Kernel performing one pass of the naive inclusive scan algorithm.
         */
        __global__ void kernNaiveScanPass(int n, int offset, int* odata, const int* idata) {
            int index = threadIdx.x + blockIdx.x * blockDim.x;
            if (index >= n) return;

            if (index >= offset) {
                odata[index] = idata[index] + idata[index - offset];
            }
            else {
                odata[index] = idata[index];
            }
        }

        /**
         * Kernel converting an inclusive scan into an exclusive scan.
         */
        __global__ void kernInclusiveToExclusive(int n, int* odata, const int* idata) {
            int index = threadIdx.x + blockIdx.x * blockDim.x;
            if (index >= n) return;

            if (index == 0) {
                odata[0] = 0;
            }
            else {
                odata[index] = idata[index - 1];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO

            int* dev_buf1 = nullptr;
            int* dev_buf2 = nullptr;

            cudaMalloc((void**)&dev_buf1, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_buf1 failed!");

            cudaMalloc((void**)&dev_buf2, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_buf2 failed!");

            cudaMemcpy(dev_buf1, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy idata to dev_buf1 failed!");

            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

            // run ilog2ceil(n) iterations of naive inclusive scan
            for (int offset = 1; offset < n; offset <<= 1) {
                kernNaiveScanPass << <fullBlocksPerGrid, blockSize >> > (n, offset, dev_buf2, dev_buf1);
                checkCUDAError("kernNaiveScanPass failed!");
                std::swap(dev_buf1, dev_buf2);
            }

            // convert inclusive scan result in dev_buf1 to exclusive scan in dev_buf2
            kernInclusiveToExclusive << <fullBlocksPerGrid, blockSize >> > (n, dev_buf2, dev_buf1);
            checkCUDAError("kernInclusiveToExclusive failed!");

            cudaMemcpy(odata, dev_buf2, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy dev_buf2 to odata failed!");

            cudaFree(dev_buf1);
            cudaFree(dev_buf2);

            timer().endGpuTimer();
        }
    }
}
