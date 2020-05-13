/*
 **********************************************
 *  CS314 Principles of Programming Languages *
 *  Spring 2020                               *
 **********************************************
 */
#include <stdio.h>
#include <stdlib.h>


//Define a graph node object
class Node {
public:
	int src, dst, weight;
	
	Node(int x, int y, int z) {
		src = x;
		dst = y;
		weight = z;
	}
};

__global__ void strongestNeighborScan_gpu(int * src, int * oldDst, int * newDst, int * oldWeight, int * newWeight, int * madeChanges, int distance, int numEdges) {
	/*YOUR CODE HERE*/
	/* The graph is encoded as an edge list consisting of three arrays: src, dst, and weight, such
	that src[n] is the source node for the n-th edge, dst[n] is the destination node for the n-th
	edge, and weight[n] is the weight of the n-th edge. The graph is undirected, so if src[n]=x
	and dst[n]=y then there exists an edge m such that src[m]=y and dst[m]=x. */

	//Get thread ID 
	int tID = blockidx.x * blockDim.x + threadidx.x;

	//Terminate if thread ID is larger than array
	if(tID >= numEdges) return;

	//Current node
	Node left(tID);

	//Get index of left node (equal to stride length)
	int rightNodeIndex = tID + distance;

	//Enforce array bound
	if(rightNodeIndex > numEdges) {
		rightNodeIndex = numEdges - 1
	};

	//Stride away node
	Node right(src[rightNodeIndex]);

	//Only compare nodes in the same stride
	if(left.src == right.src) {

		Node stronger;
		
		//Get stronger node
		if(left.weight >= right.weight) { stronger = left } else { stronger = right};

		//Set new destination
		newDst[tID] = stronger.dst;

		//Set new weight
		newWeight[tID] = stronger.weight;

		//Flag any changes
		if((newDst[tID] != oldDst[tID]) || (newWeight[tID] != oldWeight[tID])) { madeChanges = 1 };
	};
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
