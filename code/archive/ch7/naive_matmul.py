import torch

def naive_matmul(A, B):
    N = A.size(0)
    C = torch.zeros((N, N), device='cuda')
    for i in range(N):
        for j in range(N):
            # Each dot product reads A[i,:] and B[:,j] from global memory repeatedly
            C[i, j] = (A[i, :] * B[:, j]).sum()
    return C

# Usage example (small N for demo)
if __name__ == "__main__":
    N = 128
    A = torch.ones((N, N), device='cuda', dtype=torch.float32)
    B = torch.ones((N, N), device='cuda', dtype=torch.float32)
    C = naive_matmul(A, B)
    print("Done, C[0,0] =", C[0,0].item())
