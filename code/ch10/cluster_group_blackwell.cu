// Blackwell-only example: CTA cluster cooperative groups
// Demonstrates cluster-wide synchronization and shared data exchange
// Requires SM100 and a GPU/driver supporting cooperative cluster launch

#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <cstdio>

namespace cg = cooperative_groups;

// Kernel that sums values across blocks in a cluster using block-scoped shared arrays
__global__ void cluster_sum_kernel(const float *in, float *out, int elems_per_block) {
    cg::cluster_group cluster = cg::this_cluster();
    cg::thread_block cta = cg::this_thread_block();

    extern __shared__ float sdata[]; // per-CTA shared buffer

    // Per-block partial sum
    float sum = 0.0f;
    int base = blockIdx.x * elems_per_block;
    for (int i = threadIdx.x; i < elems_per_block; i += blockDim.x) {
        sum += in[base + i];
    }

    // Reduce within block
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // First thread of each block writes its partial to global memory region indexed by cluster
    if (threadIdx.x == 0) {
        out[blockIdx.x] = sdata[0];
    }

    // Synchronize all CTAs in the cluster before a follow-up phase
    cluster.sync();

    // Optionally, one CTA in the cluster aggregates the per-CTA partials
    if (cluster.block_rank() == 0 && threadIdx.x == 0) {
        float cluster_total = 0.0f;
        int cluster_blocks = cluster.dim_blocks().x; // CTAs in the cluster
        for (int b = 0; b < cluster_blocks; ++b) {
            int global_block = cluster.block_rank() + b; // simplistic mapping
            cluster_total += out[global_block];
        }
        // Write back the cluster total to the first block's slot
        out[cluster.block_rank()] = cluster_total;
    }
}

int main() {
    int num_blocks = 8; // total CTAs
    int elems_per_block = 1 << 20; // 1M elements per block
    int threads = 256;
    size_t total_elems = size_t(num_blocks) * elems_per_block;
    size_t bytes = total_elems * sizeof(float);

    float *d_in = nullptr, *d_out = nullptr;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, num_blocks * sizeof(float));
    cudaMemset(d_in, 0, bytes);
    cudaMemset(d_out, 0, num_blocks * sizeof(float));

    // Setup cluster launch configuration (2 CTAs per cluster as an example)
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(num_blocks, 1, 1);
    cfg.blockDim = dim3(threads, 1, 1);
    cfg.sharedMem = threads * sizeof(float);

    // Cluster dimensions: group blocks into clusters of size 2
    cudaLaunchAttribute attr[2];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = 2; // 2 CTAs per cluster (tune per GPU)
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;

    attr[1].id = cudaLaunchAttributeNonPortableClusterSizeAllowed;
    attr[1].val.nonPortClusterSizeAllowed = 1; // allow non-portable sizes when supported

    cfg.attrs = attr;
    cfg.numAttrs = 2;

    cudaError_t err = cudaLaunchKernelEx(&cfg,
                                         cluster_sum_kernel,
                                         (void const **)(nullptr));

    if (err != cudaSuccess) {
        printf("Cluster launch not supported or failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_in);
        cudaFree(d_out);
        return 0; // not fatal for demonstration
    }

    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel error: %s\n", cudaGetErrorString(err));
        cudaFree(d_in);
        cudaFree(d_out);
        return -1;
    }

    printf("cluster_group_blackwell completed.\n");
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}


