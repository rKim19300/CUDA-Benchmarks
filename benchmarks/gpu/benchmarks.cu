#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<fcntl.h>
#include<time.h>
#include<sys/wait.h>
#include<signal.h>
#include<cuda.h>
#include<nvml.h>

__global__ void matrixMult(float *d_A, float *d_B, float *d_R, int dim) {

    int i = (blockDim.x * blockIdx.x) + threadIdx.x;
    int j = (blockDim.y * blockIdx.y) + threadIdx.y; 

    if ((i < dim) && (j < dim)) {
        float result = 0.0;
        for (int k = 0; k < dim; k++) {
            result += d_A[i * dim + k] * d_B[k * dim + j];
        }
        d_R[i * dim + j] = result;
    }
}

// Error checker from: https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void stopAndGetTime(float *time, cudaEvent_t start, cudaEvent_t stop);

// The below expects a NVIDIA GeForce RTX 3060 Laptop GPU
int main(void) {

    // -- Initialize the data shapes for the tests --

        int testDims[][4] = {   // Test 1
                                {32, 64,     // Grid size
                                32, 16},     // Block size
                                // Test 2
                                {128, 128,   // Grid size
                                8, 8},       // Block size
                                // Test 3
                                {32, 32,     // Grid size
                                32, 32},     // Block size     
                                // Test 4
                                {37, 37,     // Block size  
                                28, 28 }     // Block size  
                                };

    // Test names
    char testNames[][9] = {"Test 1", "Test 2", "Test 3", "Test 4"};

    // -- redirect output to files -- 
    int fdOut = open("./out/benchmarks.out", O_RDWR | O_TRUNC | O_CREAT, 0777);
    dup2(fdOut, 1); // move stdout to the file
    close(fdOut);

    // -- Initialize the matricies 
    int N = (2 << (20 - 1)); // 2^20
    int dim = (int)sqrt(N); // This will work if N is being put to an even power

    // Host matricies
    float *h_A = (float*)malloc(sizeof(float) * N);
    float *h_B = (float*)malloc(sizeof(float) * N);
    float *h_R = (float*)calloc(N, sizeof(float));

    // Fill matricies with 2
    for (int i = 0; i < N; i++) {
        h_A[i] = 2;
        h_B[i] = 2;
    }

    // Move the matricies from host to device
    float *d_A, *d_B, *d_R;

    //gpuErrchk( cudaDeviceSetLimit(cudaLimitMallocHeapSize, 4 * N * sizeof(float)) );

    gpuErrchk( cudaMalloc(&d_A, N * sizeof(float)) );
    gpuErrchk( cudaMalloc(&d_B, N * sizeof(float)) );
    gpuErrchk( cudaMalloc(&d_R, N * sizeof(float)) );

    gpuErrchk( cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_R, h_R, N * sizeof(float), cudaMemcpyHostToDevice) );

    free(h_A);
    free(h_B);

    // -- Execute the loop -- 

    // Declare timer variables
    cudaEvent_t start, stop;
    gpuErrchk( cudaEventCreate(&start) );
    gpuErrchk( cudaEventCreate(&stop) );
    float time;

    // Array for test times
    float testTotals[] = {0, 0, 0, 0};

    // Call kernel in a loop
    int numRuns = 100; 
    int numTests = sizeof(testDims[0]) / sizeof(testDims[0][0]);

    // Run all tests 
    for (int test = 0; test < numTests; test++) {

        printf("starting Test %s\n", testNames[test]);
        printf("time\n");
        fflush(stdout);

        // Unpack the test dimensions 
        int gridX = testDims[test][0];
        int gridY = testDims[test][1];
        int blockX = testDims[test][2];
        int blockY = testDims[test][3];

        // Run the tests for matrix shapes
        for (int run = 0; run < numRuns; run++) {

            // Start the timer
            gpuErrchk( cudaEventRecord(start, 0) );

            // Execute kernel
            matrixMult <<<dim3(gridX, gridY), dim3(blockX, blockY)>>> (d_A, d_B, d_R, dim);

            // stop timer
            gpuErrchk( cudaEventRecord(stop, 0) );
            gpuErrchk( cudaEventSynchronize(stop) );
            gpuErrchk( cudaEventElapsedTime(&time, start, stop) ); // in ms

            // Calculate time
            testTotals[test] += time;
            printf("%d,%f\n", test, time);
            fflush(stdout);
        }
        printf("\n");
    }

    // Find the average of each test dimension test 
    printf("Test Dimension Averages\n");
    for (int i = 0; i < numTests; i++) {
        float testAverage = testTotals[i] / numRuns;
        printf("The average for %s is %f\n", testNames[i], testAverage);
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_R);
    free(h_R);

    return 0;
}
