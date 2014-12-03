//Udacity HW 4
//Radix Sorting

//=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=//
//                                                                     //
// Author : Thomas O. Wood                                             //
//                                                                     //
// File   : student_func.cu                                            //
//                                                                     //
// Descr. : Parallel radix sort for a red-eye removal image processing //
//          program implemented in CUDA.                               //
//                                                                     //
//=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=//
#include "reference_calc.cpp"
#include "utils.h"
#include <thrust/host_vector.h>
#include <stdio.h>
#define MAXTHREADS 1024


//=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=//
// Keeping scan_kernel around just to see what I did to try to make it //
// work on a large array.                                              //
//=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=//
__global__ void scan_kernel(unsigned int* d_in,
                            unsigned int* d_out,
                            unsigned int* d_incr)
{
  extern __shared__ unsigned int temp[];
  int tid = threadIdx.x;
  int myId = threadIdx.x + blockIdx.x * blockDim.x;
  int blkId = blockIdx.x;
  int offset = 1;
  int n = 2*blockDim.x;
  // Loading in a piece of the large array into shared memory.
  temp[2*tid] = d_in[2*myId];
  temp[2*tid+1] = d_in[2*myId+1];
    
  // Up-sweep reduction on block of array accessed by these threads
  for (int d = n >> 1; d > 0; d >>= 1) {
    __syncthreads();
    if (tid < d) {
      int ai = offset*(2*tid+1)-1;
      int bi = offset*(2*tid+2)-1;
      temp[bi] += temp[ai]; }
    offset *= 2; 
  }
  // Keep track of the sum by putting it in d_incr, an array with the
  // same number of elements as there are blocks of threads used to
  // scan the large array.
  __syncthreads();
  if (tid == 0) { 
    d_incr[blkId] = temp[n-1];
    temp[n-1] = 0; 
  }
    
  for (int d = 1; d < n; d*= 2) {
    offset >>= 1;
    __syncthreads();
    if (tid < d) {
      int ai = offset*(2*tid+1)-1;
      int bi = offset*(2*tid+2)-1;
      unsigned int t = temp[ai];
      temp[ai] = temp[bi];
      temp[bi] += t; }
  }
  __syncthreads();

  // Assign the values for the block-scan back to memory to be
  // incremented according to the values in d_incr in a later step.
  d_out[2*myId] = temp[2*tid];
  d_out[2*myId+1] = temp[2*tid+1];
}


__global__ void make_vectors_kernel(unsigned int* const d_in,
				    unsigned int* d_b,
				    unsigned int* d_e,
				    unsigned int mask,
				    unsigned int maskOffset,
				    const size_t numElems)
{
  int myId = threadIdx.x + blockIdx.x * blockDim.x;
  if (myId >= numElems) return;
  unsigned int bin = (d_in[myId] & mask) >> maskOffset;
  unsigned int notBin = 1 - bin;
  __syncthreads();
  d_b[myId] = bin;
  d_e[myId] = notBin;
}




//=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=//
// Used to copy results from one array to another. The parameter       //
// numElems can be used as a cutoff if the src array is bigger than    //
// the dst array and we want only the first numElems elements in our   //
// new array.                                                          //
//=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=//
__global__ void copy_kernel(unsigned int* d_in,
			    unsigned int* d_out,
			    const size_t numElems)
{
  int myId = threadIdx.x + blockIdx.x * blockDim.x;
  if (myId >= numElems) return;
  d_out[myId] = d_in[myId];  
}


//=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=//
// The kernel that increments each of the elements in the i-th block   //
// by the value stored in d_incr[i]                                    //
//=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=//
__global__ void increment_kernel(unsigned int* d_in,
				 unsigned int* d_incr,
				 unsigned int* d_out,
				 const size_t numElems)
{
  int myId = threadIdx.x + blockIdx.x * blockDim.x;
  int blkId = blockIdx.x;

  if (myId >= numElems) return;
  d_out[2*myId] = d_in[2*myId] + d_incr[blkId];
  d_out[2*myId+1] = d_in[2*myId+1] + d_incr[blkId];
}

__global__ void define_offsets_kernel(unsigned int* d_e,
				unsigned int* d_eOut,
				unsigned int* d_bOut,
				unsigned int* d_out,
				const size_t eSum,
				const size_t numElems)
{
  int myId = threadIdx.x + blockIdx.x * blockDim.x;
  
  if (myId >= numElems) return;
  
  if (d_e[myId])
    {
      d_out[myId] = d_eOut[myId];
    }
  else
    {
      d_out[myId] = eSum + d_bOut[myId];
    }
}

//=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=//
// Used to move the addresses according to the the individual bit's    //
// offset in addition to the sum number of occurences of each bit. The //
// beginning position of each "1" bit is the exclusive prescan of the  //
// d_b array added to the total number of times the "0" bit shows up   //
// in the d_b array, the sum of the d_e array where a "1" is shown     //
// wherever a zero occurs in d_b.                                      //
//=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=//
__global__ void scatter_kernel(unsigned int* const d_inVals,
			       unsigned int* const d_inPos,
			       unsigned int* d_outVals,
			       unsigned int* d_outPos,
			       unsigned int* d_offset,
			       const size_t numElems)
{
  int myId = threadIdx.x + blockIdx.x * blockDim.x;

  if (myId >= numElems) return;
  // scatter using d_offset as addresses
  d_outVals[ d_offset[myId] ] = d_inVals[ myId ];
  d_outPos[ d_offset[myId] ] = d_inPos[ myId ];

}
  


//=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=//
// scan_large_array(unsigned int* d_in,                                //
//                  unsigned int* d_out,                               //
//                  unsigned int* sum)                                 //
//                                                                     //
// scan_large_array() takes in a vector of unsigned integers and gives //
// the exclusive scan of the array and the sum of the elements in the  //
// array as output.                                                    //
//=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=//

void scan_large_array(unsigned int* d_in,
		      unsigned int* d_out,
		      unsigned int* d_sum,
		      const size_t numElems)
{
  int nBlocks = numElems / (2*MAXTHREADS) + 1;
  int N = 2*MAXTHREADS*nBlocks;
  unsigned int* d_inAug;

  checkCudaErrors(cudaMalloc((void**) &d_inAug,
			     N*sizeof(unsigned int)));
  checkCudaErrors(cudaMemset(d_inAug,
			     0,
			     N*sizeof(unsigned int)));

  copy_kernel <<< 2*nBlocks, MAXTHREADS >>> (d_in,
					     d_inAug,
					     numElems);

  unsigned int* d_outInter;
  unsigned int* d_incr;
  checkCudaErrors(cudaMalloc((void**) &d_outInter,
			     N*sizeof(unsigned int)));
  checkCudaErrors(cudaMalloc((void**) &d_incr,
			     nBlocks*sizeof(unsigned int)));

  scan_kernel <<< nBlocks,
    MAXTHREADS,
    2*MAXTHREADS*sizeof(unsigned int) >>> (d_inAug, d_outInter, d_incr);

  unsigned int* d_incrAug;
  checkCudaErrors(cudaMalloc((void**) &d_incrAug,
			     2*MAXTHREADS*sizeof(unsigned int)));
  checkCudaErrors(cudaMemset(d_incrAug,
			     0,
			     2*MAXTHREADS*sizeof(unsigned int)));

  copy_kernel <<< 2, MAXTHREADS >>> (d_incr,
				     d_incrAug,
				     nBlocks);


  unsigned int* d_incrOut;
  unsigned int* d_sumInter;

  checkCudaErrors(cudaMalloc((void**) &d_incrOut,
			     2*MAXTHREADS*sizeof(unsigned int)));
  checkCudaErrors(cudaMalloc((void**) &d_sumInter,
			     1*sizeof(unsigned int)));

  scan_kernel <<< 1,
    MAXTHREADS,
   2*MAXTHREADS*sizeof(unsigned int) >>> (d_incrAug,
					   d_incrOut,
					   d_sumInter);
  unsigned int h_sumInter[1];
  checkCudaErrors(cudaMemcpy(h_sumInter,
			     d_sumInter,
			     1*sizeof(unsigned int),
			     cudaMemcpyDeviceToHost));

  unsigned int* d_incrScan;
  checkCudaErrors(cudaMalloc((void**) &d_incrScan,
			     nBlocks*sizeof(unsigned int)));

  copy_kernel <<< 1, MAXTHREADS >>> (d_incrOut,
				    d_incrScan,
				    nBlocks);

  unsigned int* d_outFinal;
  checkCudaErrors(cudaMalloc((void**) &d_outFinal,
			     N*sizeof(unsigned int)));

  increment_kernel <<< nBlocks, MAXTHREADS >>> (d_outInter,
						d_incrScan,
						d_outFinal,
						N);

  unsigned int* d_outRed;
  checkCudaErrors(cudaMalloc((void**) &d_outRed,
			     numElems*sizeof(unsigned int)));

  copy_kernel <<< 2*nBlocks, MAXTHREADS >>> (d_outFinal,
					     d_outRed,
					     numElems);

  checkCudaErrors(cudaMemcpy(d_out,
			     d_outRed,
			     numElems*sizeof(unsigned int),
			     cudaMemcpyDeviceToDevice));

  d_sum[0] = h_sumInter[0];

}

//=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=//
// The function my_sort() will be the one to call the kernels, or at   //
// least be the function that calls the functions that call the        //
// kernels.                                                            //
//=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=//

void my_sort(unsigned int* const d_inputVals,
	     unsigned int* const d_inputPos,
	     unsigned int* const d_outputVals,
	     unsigned int* const d_outputPos,
	     const size_t numElems)
{    

  const int numBits = 1;
  const int numBins = 1 << numBits;

  // We create these copies because we are going to sort d_inputVals
  // for each LSB we can get working with the unsigned ints.
  unsigned int* d_workingVals;
  unsigned int* d_workingPos;
    
  checkCudaErrors(cudaMalloc((void**) &d_workingVals, 
			     numElems*sizeof(unsigned int)));

  checkCudaErrors(cudaMalloc((void**) &d_workingPos, 
			     numElems*sizeof(unsigned int)));
    
  checkCudaErrors(cudaMemcpy(d_workingVals, 
			     d_inputVals, 
			     numElems*sizeof(unsigned int), 
			     cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpy(d_workingPos, 
			     d_inputPos, 
			     numElems*sizeof(unsigned int), 
			     cudaMemcpyDeviceToDevice));
  

  for (unsigned int i = 0; i < 8 * sizeof(unsigned int); i += numBits){

    unsigned int mask = (numBins - 1) << i;

    // Declare variables d_b and d_e
    unsigned int* d_b;
    unsigned int* d_e;

    // Allocate memory on the GPU for d_b and d_e
    checkCudaErrors(cudaMalloc((void**) &d_b, numElems*sizeof(unsigned int)));
    checkCudaErrors(cudaMalloc((void**) &d_e, numElems*sizeof(unsigned int)));
    
    // Make vectors of the inputVals 
    make_vectors_kernel <<< numElems/MAXTHREADS + 1, 
      MAXTHREADS  >>> (d_workingVals,
		       d_b,
		       d_e,
		       mask,
		       i,
		       numElems);

    // Make output for scan_large_array() function.
    unsigned int* d_eOut;
    unsigned int* d_bOut;
    checkCudaErrors(cudaMalloc((void**) &d_eOut, 
			       numElems*sizeof(unsigned int)));
    checkCudaErrors(cudaMalloc((void**) &d_bOut,
			       numElems*sizeof(unsigned int)));

    // Set aside a place to keep the sums.
    unsigned int h_eSum[1];
    unsigned int h_bSum[1]; // Never going to use this, but whatever.

    // Scan those suckers.
    scan_large_array(d_e, d_eOut, h_eSum, numElems);
    scan_large_array(d_b, d_bOut, h_bSum, numElems);
    
    unsigned int* d_offset;
    checkCudaErrors(cudaMalloc((void**) &d_offset,
			       numElems*sizeof(unsigned int)));

    define_offsets_kernel <<< numElems/MAXTHREADS + 1, 
      MAXTHREADS >>> (d_e,
		      d_eOut,
		      d_bOut,
		      d_offset,
		      h_eSum[0],
		      numElems);

    // Get ready to launch the scatter kernel that does the sorting.
    unsigned int* d_outVals;
    unsigned int* d_outPos;
    
    checkCudaErrors(cudaMalloc((void**) &d_outVals,
			       numElems*sizeof(unsigned int)));
    checkCudaErrors(cudaMalloc((void**) &d_outPos,
			       numElems*sizeof(unsigned int)));

    scatter_kernel <<< numElems/MAXTHREADS + 1,
      MAXTHREADS >>> ( d_workingVals,
		       d_workingPos,
		       d_outVals,
		       d_outPos,
		       d_offset,
		       numElems);



    copy_kernel <<< numElems/MAXTHREADS + 1,
      MAXTHREADS >>> (d_outVals,
		      d_workingVals,
		      numElems);

    copy_kernel <<< numElems/MAXTHREADS + 1,
      MAXTHREADS >>> (d_outPos,
		      d_workingPos,
		      numElems);


    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaFree(d_e));
    checkCudaErrors(cudaFree(d_b));
    checkCudaErrors(cudaFree(d_eOut));
  }

  copy_kernel <<< numElems/MAXTHREADS + 1,
    MAXTHREADS >>> (d_workingVals,
		    d_outputVals,
		    numElems);

  copy_kernel <<< numElems/MAXTHREADS + 1,
    MAXTHREADS >>> (d_workingPos,
		    d_outputPos,
		    numElems);

  
  
}

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
	       const size_t numElems)
{
  my_sort(d_inputVals,d_inputPos,d_outputVals,d_outputPos,numElems);
}
