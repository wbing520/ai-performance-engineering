NVCC = nvcc
CFLAGS = -std=c++17 -O3

all: add

add: add.cu
	$(NVCC) $(CFLAGS) -o $@ $<

clean:
	rm -f add
