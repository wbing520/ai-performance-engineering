import torch
import torch.nn as nn
import torch.optim as optim
import time

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(1024, 512)
        self.linear2 = nn.Linear(512, 1024)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x

def main():
    device = torch.device("cuda")
    model = SimpleModel().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # Create input data
    data = torch.randn(64, 1024, device=device)
    target = torch.randn(64, 1024, device=device)

    # Warm-up run
    for _ in range(5):
        optimizer.zero_grad()
        output = model(data)
        loss = nn.MSELoss()(output, target)
        loss.backward()
        optimizer.step()

    # Regular execution timing
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(100):
        optimizer.zero_grad()
        output = model(data)
        loss = nn.MSELoss()(output, target)
        loss.backward()
        optimizer.step()
    
    torch.cuda.synchronize()
    regular_time = (time.time() - start) * 1000

    # CUDA Graph execution
    model.train()
    optimizer.zero_grad()
    
    # Capture graph
    with torch.cuda.amp.autocast():
        with torch.cuda.stream(torch.cuda.Stream()):
            static_input = data.clone()
            static_target = target.clone()
            
            # Warm-up
            output = model(static_input)
            loss = nn.MSELoss()(output, static_target)
            loss.backward()
            optimizer.step()
            
            # Capture the graph
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                optimizer.zero_grad()
                output = model(static_input)
                loss = nn.MSELoss()(output, static_target)
                loss.backward()
                optimizer.step()

    # Execute graph
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(100):
        g.replay()
    
    torch.cuda.synchronize()
    graph_time = (time.time() - start) * 1000

    print("PyTorch CUDA Graph captured")
    print(f"Graph execution time: {graph_time:.1f} ms")
    print(f"Regular execution time: {regular_time:.1f} ms")
    print(f"Memory usage: {torch.cuda.memory_allocated() / 1024 / 1024:.0f} MB")
    print(f"Speedup: {regular_time / graph_time:.1f}x")

if __name__ == "__main__":
    main()
