import torch

def tiled_matmul(A, B, tile_size=32):
    N = A.size(0)
    C = torch.zeros((N, N), device=A.device)
    # Loop over tiles
    for i in range(0, N, tile_size):
        for j in range(0, N, tile_size):
            # Compute C block (i:i+tile, j:j+tile)
            A_block = A[i:i+tile_size, :]
            B_block = B[:, j:j+tile_size]
            # Use PyTorch's optimized matrix multiply for the tile
            C[i:i+tile_size, j:j+tile_size] += torch.mm(A_block, B_block)
    return C

# Usage example
if __name__ == "__main__":
    N = 128
    A = torch.ones((N, N), device='cuda', dtype=torch.float32)
    B = torch.ones((N, N), device='cuda', dtype=torch.float32)
    C = tiled_matmul(A, B, tile_size=32)
    print("Done, C[0,0] =", C[0,0].item())
