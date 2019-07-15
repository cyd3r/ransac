CC=nvcc

all: gpu host

gpu: src/cuda.cu
	$(CC) -o gpu.out -DUSE_GPU=1 src/cuda.cu

host: src/cuda.cu
	$(CC) -o host.out -DUSE_GPU=0 src/cuda.cu

clean:
	rm -f *.out
cleanfig:
	rm -f *.jpg
cleandat:
	rm -f *.csv *.txt
