/*
 **********************************************
 *  CS314 Principles of Programming Languages *
 *  Spring 2020                               *
 **********************************************
 */
#include <stdio.h>
#include <stdlib.h>

__global__ void collateSegments_gpu(int * src, int * scanResult, int * output, int numEdges) {
	/*YOUR CODE HERE*/
	for(int i = 0; i < numEdges; i++) {
		char output = scanResult[i];
		printf("Value of src[%d] = %c \n", i, output);
	}
}
