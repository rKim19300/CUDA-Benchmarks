## NOTE

- This project requires a linux environment to run.

## Making the files

- In the root directory type "make all".

- The files executes will be compiled 
into the same file in which the source file is 
located, thus you must navigate to that directory and then type ./{filename} to run it.

### FILES

- benchmarks/cpu/query/query_cpu.c

    - Prints The specifications of the CPU
    on the machine that runs it.

- benchmarks/gpu/query/query_gpu.cu

    - Prints the specifications of the 
    GPU(s) connected to the host system if there are any.

- benchmarks/gpu/out/benchmarks.out

    - Where the output of benchmarks.cu is printed.

- benchmarks/gpu/benchmarks.cu

    - Runs the benchmarks with different grid and block dimension sizes. 

    - The original tests were ran on a "NVIDIA GeForce RTX 3060 Laptop GPU"

        - Test 1: This test utilized the maximum number of threads per block. The grid size was set to 512 (i.e. 1536 mod 512 = 0), or (32 x 16), and divided  the data size by the block size so that it perfectly fit the data with one thread per array index, grid size = (2^20)/512 = 32 x 64.

        - Test 2: This test utilized both the maximum block size and the maximum blocks per SM.  Which was block size = (1024/16) = 64 or 8 x 8 and grid size = (2^20)/64 = 128 x 128.

        - Test 3: This test makes use of the maximum block size per SM, but it won’t make use of the maximum threads per SM, because no more than one block with 1024 threads will fit the 1536 max threads in the SM. block size = (1024 or 32 x 32).  grid size = (2^20/1024) = 32 x 32.

        - Test 4: Test 4 This test choose random numbers for the block and grid size, being block size =28 x 28  and grid size=37 x 37
