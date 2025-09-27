import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_architecture():
    """Detect and return the current GPU architecture."""
    if not torch.cuda.is_available():
        return "cpu"

    device_props = torch.cuda.get_device_properties(0)
    compute_capability = f"{device_props.major}.{device_props.minor}"
    return "blackwell" if compute_capability == "10.0" else "other"


def get_architecture_info():
    """Get detailed architecture information."""
    arch = get_architecture()
    if arch == "blackwell":
        return {
            "name": "Blackwell B200/B300",
            "compute_capability": "10.0",
            "sm_version": "sm_100",
            "memory_bandwidth": "8.0 TB/s",
            "tensor_cores": "5th Gen",
            "features": ["HBM3e", "TMA", "NVLink-C2C"]
        }
    return {
        "name": "Other",
        "compute_capability": "Unknown",
        "sm_version": "Unknown",
        "memory_bandwidth": "Unknown",
        "tensor_cores": "Unknown",
        "features": []
    }

def main():
    # Set up model and data
    model_name = "deepseek-ai/DeepSeek-V3"

    # Use a tiny model so the example fits comfortably on a single GPU
    model_name = "sshleifer/tiny-gpt2"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    batch_size = 2
    input_texts = ["DeepSeek is great."] * batch_size
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    enc = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True)
    input_ids = enc.input_ids.to(device)
    attention_mask = enc.attention_mask.to(device)
    labels = input_ids.clone()  # For LM training, labels are the inputs
    
    # Warm-up (not profiled)
    print("Running warm-up iterations...")
    for _ in range(1):
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    
    print("Starting profiling...")
    
    # Profile one training iteration
    with profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        with profiler.record_function("training_step"):
            optimizer.zero_grad()
            
            with profiler.record_function("forward"):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
            
            with profiler.record_function("backward"):
                loss.backward()
            
            with profiler.record_function("optimizer_step"):
                optimizer.step()
    
    # Export traces
    prof.export_chrome_trace("deepseek_v3_trace.json")
    
    # Print profiler summary
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    
    # Export for HTA analysis (only if directory exists or create it)
    trace_dir = "./hta_traces"
    os.makedirs(trace_dir, exist_ok=True)
    try:
        prof.export_chrome_trace(f"{trace_dir}/rank_0.json")
    except RuntimeError as e:
        if "Trace is already saved" in str(e):
            print("Trace already exported, skipping duplicate export")
        else:
            raise e
    
    # Memory profiling
    if hasattr(torch.cuda, 'memory_stats'):
        memory_stats = torch.cuda.memory_stats()
        print(f"Peak memory allocated: {memory_stats.get('allocated_bytes.all.peak', 0) / 1e9:.2f} GB")
        print(f"Peak memory reserved: {memory_stats.get('reserved_bytes.all.peak', 0) / 1e9:.2f} GB")

def analyze_trace():
    """Load and analyze the generated trace file."""
    try:
        with open("deepseek_v3_trace.json", "r") as f:
            trace_data = json.load(f)
        
        print(f"Trace contains {len(trace_data.get('traceEvents', []))} events")
        
        # Find longest running events
        events = trace_data.get('traceEvents', [])
        duration_events = [e for e in events if 'dur' in e and e['dur'] > 1000]  # > 1ms
        duration_events.sort(key=lambda x: x['dur'], reverse=True)
        
        print("\nTop 10 longest running events:")
        for i, event in enumerate(duration_events[:10]):
            print(f"{i+1}. {event.get('name', 'unknown')}: {event['dur']/1000:.2f} ms")
            
    except FileNotFoundError:
        print("Trace file not found. Run the profiling first.")

if __name__ == "__main__":
    main()
    analyze_trace()

# Architecture-specific optimizations
if torch.cuda.is_available():
    device_props = torch.cuda.get_device_properties(0)
    compute_capability = f"{device_props.major}.{device_props.minor}"

    inductor = getattr(torch, "_inductor", None)
    triton_cfg = getattr(getattr(inductor, "config", None), "triton", None) if inductor else None

    if compute_capability == "10.0" and triton_cfg is not None:  # Blackwell B200/B300
        try:
            if hasattr(triton_cfg, "use_blackwell_optimizations"):
                triton_cfg.use_blackwell_optimizations = True
            if hasattr(triton_cfg, "hbm3e_optimizations"):
                triton_cfg.hbm3e_optimizations = True
            if hasattr(triton_cfg, "tma_support"):
                triton_cfg.tma_support = True
            if hasattr(triton_cfg, "stream_ordered_memory"):
                triton_cfg.stream_ordered_memory = True
        except AttributeError:
            print("Blackwell optimizations not available in this PyTorch build")

    if triton_cfg is not None and hasattr(triton_cfg, "unique_kernel_names"):
        triton_cfg.unique_kernel_names = True
    if hasattr(torch, "_dynamo") and hasattr(torch._dynamo, "config"):
        torch._dynamo.config.automatic_dynamic_shapes = True
