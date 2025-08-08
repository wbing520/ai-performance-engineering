# Chapter 1: Introduction and AI System Overview

This directory contains code examples that demonstrate the core concepts from Chapter 1 of the AI Performance Engineering book.

## Key Concepts Demonstrated

### 1. Goodput Measurement
- Measuring useful throughput vs. raw throughput
- Identifying overhead and inefficiencies
- Calculating efficiency percentages

### 2. Hardware-Software Co-Design (Mechanical Sympathy)
- Understanding how hardware capabilities influence software design
- Optimizing for specific hardware characteristics
- Measuring the impact of different configurations

### 3. Performance Profiling
- Using PyTorch profiler to identify bottlenecks
- System resource monitoring
- Performance benchmarking

## Files

- `performance_basics.py`: Main demonstration script showing goodput measurement and mechanical sympathy principles
- `requirements.txt`: Required Python dependencies
- `README.md`: This file

## Running the Examples

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the performance demonstration:
```bash
python performance_basics.py
```

## Expected Output

The script will demonstrate:
- Baseline performance measurement of a simple transformer model
- Goodput calculation and efficiency metrics
- System resource utilization (CPU, memory, GPU)
- Performance profiling insights
- Mechanical sympathy demonstration with different batch sizes

## Key Takeaways

1. **Measure Goodput**: Focus on useful work completed, not just raw throughput
2. **Hardware Awareness**: Design software with hardware capabilities in mind
3. **Profile-Driven Optimization**: Use data to guide performance improvements
4. **System-Level Thinking**: Consider the entire stack, not just individual components
5. **Small Changes, Big Impact**: Minor optimizations can have outsized effects at scale

## Dependencies

- PyTorch 2.8+ for deep learning functionality
- psutil for system monitoring
- GPUtil for GPU metrics
- numpy for numerical operations

This chapter sets the foundation for understanding AI systems performance engineering principles that will be explored in detail throughout the rest of the book.
