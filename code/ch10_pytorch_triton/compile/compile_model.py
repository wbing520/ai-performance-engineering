import torch
import torch.nn as nn
import torch.optim as optim

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1024, 1024)

    def forward(self, x):
        return self.fc(x)

def train():
    model = SimpleModel().cuda()
    model = torch.compile(model, mode="max-autotune")
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    data = torch.randn(64, 1024).cuda()
    target = torch.randn(64, 1024).cuda()
    for _ in range(50):
        optimizer.zero_grad()
        output = model(data)
        loss = nn.MSELoss()(output, target)
        loss.backward()
        optimizer.step()

if __name__ == "__main__":
    train()
