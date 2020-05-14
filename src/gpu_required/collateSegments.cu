/*
 **********************************************
 *  CS314 Principles of Programming Languages *
 *  Spring 2020                               *
 **********************************************
 */
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

using namespace std;

__global__ void collateSegments_gpu(int * src, int * scanResult, int * output, int numEdges) {
	/*YOUR CODE HERE*/
	for(int i = 0; i < numEdges; i++) {
		cout << "Value of src[" << i << "] = ";
		cout << src[i] << endl;
	}
}
