#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>

void printDevProp(cudaDeviceProp devProp);

int main(void) {

    int devCount;
    cudaGetDeviceCount(&devCount);
    printf("You have %d device(s) on your system\n\n", devCount);
    for (int i = 0; i < devCount; i++) {
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, i);
        printDevProp(devProp);
    }

    return 0;
}


void printDevProp(cudaDeviceProp devProp) {

    char dims[] = {'x', 'y', 'z'};

    printf("Major revision number: %d\n", devProp.major);
    printf("Minor Revision Number: %d\n", devProp.minor);
    printf("Name: %s\n", devProp.name);
    printf("Total global memory: %zu Bytes\n", devProp.totalGlobalMem); 
    printf("Total shared memory per block: %zu Bytes\n", devProp.sharedMemPerBlock);
    printf("Total registers per block: %d\n", devProp.regsPerBlock); // 32-bit regs
    printf("Total registers per multiprocessor: %d\n", devProp.regsPerMultiprocessor); // 32-bit regs
    printf("Warp size: %d threads\n", devProp.warpSize); 
    printf("Maximum memory pitch: %zu Bytes\n", devProp.memPitch); // max pitch allowed for mem copies
    printf("Maximum threads per block: %d\n", devProp.maxThreadsPerBlock);
    for (int i = 0; i < 3; i++) 
        printf("Maximum dimension %c of block: %d\n", dims[i], devProp.maxThreadsDim[i]);
    for (int i = 0; i < 3; i++)
        printf("Maximum dimension %c of grid: %d\n", dims[i], devProp.maxGridSize[i]);
    printf("clock rate: %d kHz\n", devProp.clockRate); // In kilohertz
    printf("Total constant memory: %zu Bytes\n", devProp.totalConstMem); 
    printf("Texture Alignment: %zu\n", devProp.textureAlignment);
    printf("Concurrent copy and execution: %s\n", ((devProp.deviceOverlap) ? "Yes" : "No"));
    printf("Number of multiprocessors: %d\n", devProp.multiProcessorCount); // SMs
    printf("Max blocks per multiprocessor %d\n", devProp.maxBlocksPerMultiProcessor);
    printf("Max threads per multiprocessor %d\n", devProp.maxThreadsPerMultiProcessor);
}

