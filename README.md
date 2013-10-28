cudaUVAPing
===========

Benchmarks of PING of using CUDA UVA-access with permanent scan (CUDA 5.5 and C++11)

Source code requires: CUDA >= 4.0 and (C++11 or boost 1.53)

Recomended: CUDA 5.5 and C++11 in MSVS 2012 with Nsight VS 3.1

Comparison PING - latencies for memory access of small blocks **1 Byte**:
- standard MPP-approach with using DMA-controller and launching kernel-function for each block of data (good for block_size >= 1 MB)
- permanent scan of CPU-pinned-memory for the new data by using Unified Virtual Addressing and launching kernel-function once at all time (good for block_size < 1 MB)


Result of benchmarks on GeForce GTX460 SE CC2.1 see in: result.txt