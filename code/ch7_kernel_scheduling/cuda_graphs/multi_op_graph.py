import torch, time
import torch.cuda as ctc

if __name__ == "__main__":
    x = torch.zeros(1<<20, device='cuda')
    g = ctc.CUDAGraph()
    stream = torch.cuda.current_stream()
    stream.wait_stream(torch.cuda.default_stream())
    g.capture_begin()
    x.add_(1)
    g.capture_end()
    torch.cuda.synchronize()
    t0 = time.time()
    g.replay()
    torch.cuda.synchronize()
    print("Graph time:", time.time() - t0)
