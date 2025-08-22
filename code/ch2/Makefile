# Chapter 2 Makefile (includes Blackwell-only NVLink C2C demo)

ARCH ?= sm_90
CUDA_VERSION = 12.8
NVCC = nvcc
NVCC_FLAGS = -O3 -std=c++17 -arch=$(ARCH) --expt-relaxed-constexpr -DCUDA_VERSION=$(CUDA_VERSION)

TARGETS =

ifeq ($(ARCH),sm_100)
TARGETS += nvlink_c2c_p2p_blackwell
endif

all: $(TARGETS)
	@echo "Building ch2 targets with $(ARCH) (CUDA $(CUDA_VERSION))"

nvlink_c2c_p2p_blackwell: nvlink_c2c_p2p_blackwell.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

clean:
	rm -f $(TARGETS) *.o *.so *.a

.PHONY: all clean $(TARGETS)

