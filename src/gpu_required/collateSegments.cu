/*
 **********************************************
 *  CS314 Principles of Programming Languages *
 *  Spring 2020                               *
 **********************************************
 */
#include <stdio.h>
#include <stdlib.h>

__global__ void collateSegments_gpu(int * src, int * scanResult, int * output, int numEdges) {
	//Get thread ID 
	int tID = blockIdx.x * blockDim.x + threadIdx.x;

	//Terminate if thread ID is larger than array
	//if(tID > numEdges - 1) return;

	//Check if at end of segment
	if(src[tID] != src[tID+1]) {
		output[src[tID]] = scanResult[tID];
	}

	//Edge Case
	if(tID = numEdges - 1) {
		output[src[tID]] = scanResult[tID];
	}
}
