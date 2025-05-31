import torch
from torch.utils.checkpoint import checkpoint

# Example activation checkpointing
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(1024, 1024)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(1024, 1024)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)

def train():
    model = Block().cuda()
    data = torch.randn(64, 1024).cuda()
    # Wrap the second half in checkpoint
    def block2(x):
        return model.fc2(model.relu(model.fc1(x)))
    x = checkpoint(block2, data)
    print("Checkpoint output:", x.mean().item())

if __name__ == "__main__":
    train()
