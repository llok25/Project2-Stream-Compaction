#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();

        void scan(int n, int *odata, const int *idata);

        int compact(int n, int *odata, const int *idata);

        // Device-to-device scan for internal use by Radix Sort
        void runWorkEfficientScan(int paddedN, int* dev_data);
    }
}
