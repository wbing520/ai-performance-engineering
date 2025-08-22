#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

static bool checkPeerAccess(int devA, int devB) {
	int canAccess = 0;
	cudaDeviceCanAccessPeer(&canAccess, devA, devB);
	return canAccess != 0;
}

static float measureP2P(int devSrc, int devDst, size_t bytes, int iters) {
	cudaSetDevice(devSrc);
	void *src = nullptr;
	cudaMalloc(&src, bytes);
	cudaMemset(src, 0, bytes);

	cudaSetDevice(devDst);
	void *dst = nullptr;
	cudaMalloc(&dst, bytes);
	cudaMemset(dst, 0, bytes);

	cudaStream_t stream;
	cudaStreamCreate(&stream);

	// Warmup
	for (int i = 0; i < 3; ++i) {
		cudaMemcpyPeerAsync(dst, devDst, src, devSrc, bytes, stream);
	}
	cudaStreamSynchronize(stream);

	// Timed iterations
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, stream);
	for (int i = 0; i < iters; ++i) {
		cudaMemcpyPeerAsync(dst, devDst, src, devSrc, bytes, stream);
	}
	cudaEventRecord(stop, stream);
	cudaEventSynchronize(stop);

	float ms = 0.0f;
	cudaEventElapsedTime(&ms, start, stop);
	float avgMs = ms / iters;
	float gbps = (bytes / 1e9f) / (avgMs / 1e3f);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaStreamDestroy(stream);

	cudaFree(dst);
	cudaSetDevice(devSrc);
	cudaFree(src);

	return gbps;
}

int main() {
	int deviceCount = 0;
	cudaGetDeviceCount(&deviceCount);
	if (deviceCount < 2) {
		printf("Need at least 2 GPUs for NVLink C2C demo. Found %d\n", deviceCount);
		return 0;
	}

	// Enable peer access where possible
	for (int i = 0; i < deviceCount; ++i) {
		for (int j = 0; j < deviceCount; ++j) {
			if (i == j) continue;
			if (checkPeerAccess(i, j)) {
				cudaSetDevice(i);
				cudaDeviceEnablePeerAccess(j, 0);
			}
		}
	}

	// Test large transfers to highlight Blackwell NVLink-C2C capability
	size_t bytes = size_t(1) << 30; // 1 GiB
	int iters = 10;

	for (int src = 0; src < deviceCount; ++src) {
		cudaDeviceProp propS{};
		cudaGetDeviceProperties(&propS, src);
		for (int dst = 0; dst < deviceCount; ++dst) {
			if (src == dst) continue;
			cudaDeviceProp propD{};
			cudaGetDeviceProperties(&propD, dst);
			if (!checkPeerAccess(src, dst)) {
				printf("%s -> %s: Peer access not available\n", propS.name, propD.name);
				continue;
			}
			float gbps = measureP2P(src, dst, bytes, iters);
			printf("%s -> %s: %.2f GB/s (1 GiB avg over %d iters)\n", propS.name, propD.name, gbps, iters);
		}
	}

	return 0;
}


