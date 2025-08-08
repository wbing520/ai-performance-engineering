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
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler()
    data = torch.randn(64, 1024).cuda()
    target = torch.randn(64, 1024).cuda()
    for _ in range(50):
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            output = model(data)
            loss = nn.MSELoss()(output, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

if __name__ == "__main__":
    train()
