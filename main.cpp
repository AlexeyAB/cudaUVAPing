#include <iostream>
#include <ctime>       // clock_t, clock, CLOCKS_PER_SEC

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "kernel.h"

// works since CUDA 5.5, Nsight VS 3.1 which works with MSVS 2012 C++11
#if __cplusplus >= 201103L || _MSC_VER >= 1700
	#include <thread>
	using std::thread;
#else
	#include <boost/thread.hpp>
	using boost::thread;
#endif


int main() {

	srand (time(NULL));

	// count devices & info
	int device_count;
	cudaDeviceProp device_prop;

	// get count Cuda Devices
	cudaGetDeviceCount(&device_count);
	std::cout << "Device count: " <<  device_count << std::endl;

	if (device_count > 100) device_count = 0;
	for (int i = 0; i < device_count; i++)
	{
		// get Cuda Devices Info
		cudaGetDeviceProperties(&device_prop, i);
		std::cout << "Device" << i << ": " <<  device_prop.name;
		std::cout << " (" <<  device_prop.totalGlobalMem/(1024*1024) << " MB)";
		std::cout << ", CUDA capability: " <<  device_prop.major << "." << device_prop.minor << std::endl;	
		std::cout << "UVA: " <<  device_prop.unifiedAddressing << std::endl;
		std::cout << "UVA - canMapHostMemory: " <<  device_prop.canMapHostMemory  << std::endl;
		std::cout << "DMA - asyncEngineCount: " <<  device_prop.asyncEngineCount << std::endl;
		std::cout << "MAX BLOCKS NUMBER: " <<  device_prop.maxGridSize[0] << std::endl;
	}
	std::cout << std::endl;
	std::cout << "BLOCKS_NUMBER = 1, THREADS_NUMBER = 32" << std::endl;
	//std::cout << "__cplusplus = " << __cplusplus << std::endl;	
	std::cout << std::endl;
	std::cout << "DMA - copy by using DMA-controller and launch kernel-function for each packet" << std::endl;
	std::cout << "UVA - copy by using Unified Virtual Addressing and launch kernel-function once" << std::endl;
	std::cout << "--------------------------------------------------------------------" << std::endl;
	std::cout << std::endl;

	// Can Host map memory
	cudaSetDeviceFlags(cudaDeviceMapHost);


	// init pointers

	// src: temp buffer & flag
	unsigned char * host_src_buff_ptr = NULL;
	bool * host_src_flag_ptr = NULL;

	// dst: temp buffer & flag
	unsigned char * host_dst_buff_ptr = NULL;
	bool * host_dst_flag_ptr = NULL;
	
	// temp device memory
	unsigned char * dev_src_ptr1 = NULL;
	unsigned char * dev_src_ptr2 = NULL;
	unsigned char * dev_dst_ptr1 = NULL;
	unsigned char * dev_dst_ptr2 = NULL;



	clock_t end, start;

	// sizes of buffer and flag
	static const unsigned int c_buff_size = 4096;			// must not be large than 16384
	static const unsigned int c_flags_number = 1;


	// Allocate memory
	cudaHostAlloc(&host_src_buff_ptr, c_buff_size, cudaHostAllocMapped | cudaHostAllocPortable );
	cudaHostAlloc(&host_src_flag_ptr, sizeof(*host_dst_flag_ptr)*c_flags_number, cudaHostAllocMapped | cudaHostAllocPortable );
	for(size_t i = 0; i < c_flags_number; ++i) 
		host_src_flag_ptr[i] = false;

	cudaHostAlloc(&host_dst_buff_ptr, c_buff_size, cudaHostAllocMapped | cudaHostAllocPortable );
	cudaHostAlloc(&host_dst_flag_ptr, sizeof(*host_dst_flag_ptr)*c_flags_number, cudaHostAllocMapped | cudaHostAllocPortable );
	for(size_t i = 0; i < c_flags_number; ++i) 
		host_dst_flag_ptr[i] = false;

	cudaMalloc(&dev_src_ptr1, c_buff_size);
	cudaMalloc(&dev_src_ptr2, c_buff_size);
	cudaMalloc(&dev_dst_ptr1, c_buff_size);
	cudaMalloc(&dev_dst_ptr2, c_buff_size);

	// create none-zero stream
	cudaStream_t stream1;
	cudaStreamCreate(&stream1);

	std::cout << "Test sequential: memory copy H->D & D->H and launch of kernel-function:" << std::endl;

	// =============================================================
	// Test DMA Ping-Pong
		static const unsigned int iters_dma = 100*1000;
		cudaDeviceSynchronize();

		start = clock();
		for(size_t i = 0; i < iters_dma; ++i) {
			cudaMemcpy(dev_src_ptr1, host_src_buff_ptr, 1, cudaMemcpyDefault);	

			k_gpu_ram(dev_dst_ptr1, dev_src_ptr1, 1, 32, stream1);
			cudaDeviceSynchronize();

			cudaMemcpy(host_dst_buff_ptr, dev_dst_ptr1, 1, cudaMemcpyDefault);
		}
		end = clock();

		const float c_time_memcpy = (float)(end - start)/(CLOCKS_PER_SEC);
		std::cout << "DMA Memcpy & Kernel: iters = " << iters_dma << 
			", \t time: " << c_time_memcpy << 
			", ping = " << c_time_memcpy/iters_dma << " sec" << std::endl;

		

	// =============================================================
	// Test UVA Ping-Pong
		static const unsigned int iters_uva = 1*1000*1000;

		// src: temp buffer & flag
		unsigned char * uva_src_buff_ptr = NULL;
		bool * uva_src_flag_ptr = NULL;

		// dst: temp buffer & flag
		unsigned char * uva_dst_buff_ptr = NULL;
		bool * uva_dst_flag_ptr = NULL;
		
		// host_ptr -> uva_ptr
		cudaHostGetDevicePointer(&uva_src_buff_ptr, host_src_buff_ptr, 0);
		cudaHostGetDevicePointer(&uva_src_flag_ptr, host_src_flag_ptr, 0);

		cudaHostGetDevicePointer(&uva_dst_buff_ptr, host_dst_buff_ptr, 0);		
		cudaHostGetDevicePointer(&uva_dst_flag_ptr, host_dst_flag_ptr, 0);	



		const bool init_flag = const_cast<volatile bool *>(host_src_flag_ptr)[0] = false;
		cudaDeviceSynchronize();

		thread t1( [&] () {
			k_spin_test(uva_dst_flag_ptr, uva_src_flag_ptr, uva_dst_buff_ptr, uva_src_buff_ptr, init_flag, iters_uva, 1, 32, stream1);
			cudaDeviceSynchronize();
		} );

		start = clock();
		for(size_t i = 0; i < iters_uva; ++i) {
			//std::cout << i << std::endl;
			// set query flag
			const_cast<volatile bool *>(host_src_flag_ptr)[0] = !const_cast<volatile bool *>(host_src_flag_ptr)[0];
			
			// wait answer flag
			while (const_cast<volatile bool *>(host_dst_flag_ptr)[0] != const_cast<volatile bool *>(host_src_flag_ptr)[0]) {
			}
		}
		end = clock();

		const float c_time_ping_pong = (float)(end - start)/(CLOCKS_PER_SEC);
		std::cout << "UVA PING-PONG: iters = " << iters_uva << 
			", \t time: " << c_time_ping_pong << 
			", ping = " << c_time_ping_pong/iters_uva << " sec" << std::endl;
		
		t1.join();

		std::cout << "UVA faster than DMA in: " << (c_time_memcpy/iters_dma) / (c_time_ping_pong/iters_uva) <<
			" X times" << std::endl;



	int b;
	std::cin >> b;

	return 0;
}