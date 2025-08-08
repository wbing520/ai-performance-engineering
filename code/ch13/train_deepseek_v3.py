# train_deepseek_v3.py
import torch
import torch.profiler as profiler
from torch.profiler import ProfilerActivity
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

def main():
    # Set up model and data
    model_name = "deepseek-ai/DeepSeek-V3"
    
    # For this example, we'll use a smaller model for demonstration
    # Replace with actual DeepSeek-V3 when available
    model_name = "microsoft/DialoGPT-medium"  # Fallback for demo
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    batch_size = 4
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
    for _ in range(5):
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
        with_stack=True,
        use_cuda=True
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
    
    # Export for HTA analysis
    trace_dir = "./hta_traces"
    prof.export_chrome_trace(f"{trace_dir}/rank_0.json")
    
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
