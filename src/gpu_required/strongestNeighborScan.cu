/*
 **********************************************
 *  CS314 Principles of Programming Languages *
 *  Spring 2020                               *
 **********************************************
 */

 //	       ./gsn ./testcases/input5.mtx output.txt 200

#include <stdio.h>
#include <stdlib.h>

__global__ void strongestNeighborScan_gpu(int * src, int * oldDst, int * newDst, int * oldWeight, int * newWeight, int * madeChanges, int distance, int numEdges) {
	/*YOUR CODE HERE*/
	/* The graph is encoded as an edge list consisting of three arrays: src, dst, and weight, such
	that src[n] is the source node for the n-th edge, dst[n] is the destination node for the n-th
	edge, and weight[n] is the weight of the n-th edge. The graph is undirected, so if src[n]=x
	and dst[n]=y then there exists an edge m such that src[m]=y and dst[m]=x. */

	//Get total num of threads
	int totalThreads = blockDim.x * gridDim.x;

	/*
		numEdges = 1014
		totalThreads = 256
		/
		factnor = 4

		thus for each tID, add self and repeat, n
	*/
	//Get numOfJobs
	int numOfJobs = Math.floor(totalThreads / numEdges);

	//Get thread ID 
	int tID = blockIdx.x * blockDim.x + threadIdx.x;

	for(let i = 1; i <= numOfJobs; i++) {
		tID *= i;

		//Terminate if thread ID is larger than array
		if(tID >= numEdges) return;

		//Current node
		int rightIndex = tID;

		//Stride away node
		int leftIndex = tID - distance;

		//Enforce array bound
		if(leftIndex < 0) { leftIndex = 0; };


		//Only compare nodes in the same segment
		if(src[leftIndex] == src[rightIndex]) {
			int strongerIndex;
			
			//Get stronger node
			if(oldWeight[leftIndex] > oldWeight[rightIndex]) { 
				strongerIndex = leftIndex; 
			} else if(oldWeight[leftIndex] < oldWeight[rightIndex]){ 
				strongerIndex = rightIndex; 
			} else {
				//if equal weights, take node with smaller vID
				if(oldDst[leftIndex] < oldDst[rightIndex]) { 
					strongerIndex = leftIndex; 
				} else { 
					strongerIndex = rightIndex; 
				};
			}

			//Set new destination
			newDst[tID] = oldDst[strongerIndex];

			//Set new weight
			newWeight[tID] = oldWeight[strongerIndex];

			//Flag any changes
			if((newDst[tID] != oldDst[tID]) || (newWeight[tID] != oldWeight[tID])) { *madeChanges = 1; };

		} else {
			//Different segments defaults to no change
			newDst[tID] = oldDst[tID];
			newWeight[tID] = oldWeight[tID];
		}
	}
}

/*
 * Performs segment scan to find strongest neighbor for each src node
 * @param src The source array in the edge list
 * @param oldDst The current dst array in the edge list
 * @param newDst The modified dst array produced by this GPU kernel function
 * @param oldWeight The current weight array in the edge list
 * @param newWeight The modified weight array produced by this GPU kernel function
 * @param madeChanges If our output is different than our input then we must set *madeChanges to 1, so the host will know to launch another step of the scan.
 * @param distance The distance between array locations being examined. This is always a power of 2.
 * @param numEdges The size of the index, weight, and flags arrays.
*/