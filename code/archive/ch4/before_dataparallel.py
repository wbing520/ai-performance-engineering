import torch
import torch.nn as nn
import torch.optim as optim
import time

# Dummy model and dataset for demonstration
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SimpleNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)
    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

# Setup model and data
input_size = 1024
hidden_size = 256
model = SimpleNet(input_size, hidden_size).cuda()  # initially on GPU0
model = nn.DataParallel(model)  # spreads across GPUs 0 and 1
optimizer = optim.SGD(model.parameters(), lr=0.01)
data = torch.randn(512, input_size).cuda()   # batch of 512 on GPU0 (DataParallel will split it)
target = torch.randn(512, 1).cuda()

# Time a single training step
torch.cuda.synchronize()
start = time.time()
output = model(data)                     # forward pass (DP splits batch across GPUs)
loss = nn.functional.mse_loss(output, target)
loss.backward()                          # backward (DP gathers grads to GPU0)
optimizer.step()
torch.cuda.synchronize()
elapsed = time.time() - start
print(f"DataParallel step took {elapsed*1000:.2f} ms")
