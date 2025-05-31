#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <stdio.h>
namespace cg=cooperative_groups;
__global__ void MyClusterKernel(){
    cg::cluster_group cluster=cg::this_cluster();
    extern __shared__ int shared_buffer[];
    int rank=cluster.thread_block_rank();
    for(int t=0;t<1;++t){
        // init
        shared_buffer[threadIdx.x]=threadIdx.x;
        cluster.sync();
        int* remote=cluster.map_shared_rank(shared_buffer,0);
        if(rank!=0) atomicAdd(&remote[0], shared_buffer[0]);
        cluster.sync();
        if(rank==0 && threadIdx.x==0)
            printf("Cluster sum: %d\n", shared_buffer[0]);
    }
}
int main(){
    cudaLaunchClusterKernel(MyClusterKernel, dim3(4), dim3(256), dim3(2), 256*sizeof(int));
    cudaDeviceSynchronize();
    return 0;
}