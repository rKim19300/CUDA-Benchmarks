CC = gcc
NVCC = nvcc
CFLAGS = -Wall -Werror
CUDAFLAGS = -lnvidia-ml

src_files_c = ./benchmarks/cpu/query/query_cpu.c

src_files_cu = ./benchmarks/gpu/query/query_gpu.cu ./benchmarks/gpu/benchmarks.cu

obj_files_c = $(src_files_c:.c=)

obj_files_cu = $(src_files_cu:.cu=)

.PHONY: clean all
all: $(obj_files_c) $(src_files_c)

%: %.c
	$(CC) -o $@ $< $(CFLAGS)

all: $(obj_files_cu) $(src_files_cu)

%: %.cu
	$(NVCC) -o $@ $< $(CUDAFLAGS)
	
clean:
	rm -f $(obj_files_c)
	rm -f $(obj_files_cu)

