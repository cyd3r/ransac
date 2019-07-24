CC=nvcc

# build binaries

all: gpu host

gpu: src/cuda.cu
	$(CC) -o gpu.out -DUSE_GPU=1 src/cuda.cu

host: src/cuda.cu
	$(CC) -o host.out -DUSE_GPU=0 src/cuda.cu

# clean commands

clean: cleanbin cleanfig cleandat

cleanbin:
	rm -f *.out */__pycache__
cleanfig:
	rm -f *.jpg
cleandat:
	rm -f *.csv *.txt

report: report/*
	cd report; pandoc -o report.pdf report.md

