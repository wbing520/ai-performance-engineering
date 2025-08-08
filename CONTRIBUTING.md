# Contributing to AI Systems Performance Engineering

Thank you for your interest in contributing to the AI Systems Performance Engineering repository! This guide will help you get started with contributing code, documentation, examples, and improvements.

## 🤝 How to Contribute

We welcome contributions from the community in many forms:

- **Code Examples**: New CUDA kernels, PyTorch optimizations, performance scripts
- **Documentation**: Improvements to README files, code comments, tutorials
- **Performance Optimizations**: Better algorithms, memory optimizations, profiling tools
- **Bug Fixes**: Issues with existing code, compatibility problems
- **Architecture Support**: Support for new GPU architectures, frameworks
- **Testing**: Unit tests, performance benchmarks, validation scripts

## 🚀 Getting Started

### Prerequisites

- NVIDIA GPU with CUDA support
- Python 3.8+
- PyTorch with CUDA
- Git

### Development Setup

```bash
# Fork and clone the repository
git clone https://github.com/your-username/ai-performance-engineering.git
cd ai-performance-engineering

# Create a new branch for your contribution
git checkout -b feature/your-feature-name

# Install development dependencies
pip install -r code/ch1/requirements.txt
```

## 📝 Contribution Guidelines

### Code Style

- **Python**: Follow PEP 8 style guidelines
- **CUDA**: Use consistent naming conventions and proper error handling
- **Shell Scripts**: Use bash with proper error handling (`set -e`)
- **Comments**: Add clear, descriptive comments for complex logic

### File Organization

- **New Examples**: Place in appropriate chapter directory (`code/chX/`)
- **Tools**: Add to `tools/` directory
- **Scripts**: Add to `code/profiler_scripts/` or relevant chapter
- **Documentation**: Update relevant README files

### Architecture Support

When adding new examples, ensure they support multiple architectures:

```bash
# Test on different architectures
./code/switch_architecture.sh sm_90  # Hopper H100/H200
./code/switch_architecture.sh sm_100 # Blackwell B200/B300
```

## 🔧 Development Workflow

### 1. Choose Your Contribution Type

#### **Code Examples**
- Create new CUDA kernels or PyTorch optimizations
- Add performance profiling scripts
- Implement new algorithms or techniques

#### **Documentation**
- Improve README files with better explanations
- Add code comments and docstrings
- Create tutorials or guides

#### **Performance Optimizations**
- Optimize existing code for better performance
- Add new profiling tools
- Improve memory usage or compute efficiency

### 2. Development Process

```bash
# Make your changes
# Test your code thoroughly

# Run tests (if applicable)
python -m pytest tests/

# Check code style
black code/
flake8 code/

# Test architecture switching
./code/switch_architecture.sh sm_90
./code/switch_architecture.sh sm_100
```

### 3. Testing Your Changes

#### **Performance Testing**
```bash
# Run performance benchmarks
./code/build_all.sh

# Profile your changes
./code/profiler_scripts/comprehensive_profile.sh

# Compare with baseline
python tools/comprehensive_profiling.py
```

#### **Compatibility Testing**
- Test on different GPU architectures
- Verify PyTorch version compatibility
- Check CUDA version compatibility

### 4. Submitting Your Contribution

```bash
# Add your changes
git add .

# Commit with descriptive message
git commit -m "Add new CUDA kernel for memory optimization

- Implements coalesced memory access pattern
- Supports both Hopper and Blackwell architectures
- Includes performance benchmarks
- Adds comprehensive documentation"

# Push to your fork
git push origin feature/your-feature-name
```

## 📋 Pull Request Guidelines

### Before Submitting

- [ ] **Test thoroughly** on multiple architectures
- [ ] **Update documentation** if needed
- [ ] **Add comments** for complex code
- [ ] **Include performance benchmarks** for optimizations
- [ ] **Follow naming conventions** and code style
- [ ] **Update relevant README files**

### Pull Request Template

```markdown
## Description
Brief description of your changes

## Type of Change
- [ ] New feature (code example, optimization)
- [ ] Bug fix
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Architecture support

## Testing
- [ ] Tested on Hopper H100/H200 (sm_90)
- [ ] Tested on Blackwell B200/B300 (sm_100)
- [ ] Performance benchmarks included
- [ ] Documentation updated

## Performance Impact
- **Before**: [baseline metrics]
- **After**: [improved metrics]
- **Improvement**: [percentage/description]

## Additional Notes
Any additional context or considerations
```

## 🏗️ Architecture Guidelines

### Adding New GPU Support

When adding support for new GPU architectures:

1. **Update architecture detection scripts**
2. **Add new architecture constants**
3. **Test on target hardware**
4. **Update documentation**

### Example Architecture Addition

```bash
# Add new architecture support
echo "sm_110" >> code/arch_config.py

# Update switch_architecture.sh
# Add new architecture to detection logic

# Test compatibility
./code/switch_architecture.sh sm_110
```

## 📊 Performance Contribution Guidelines

### Benchmarking Standards

- **Baseline**: Always include baseline performance
- **Multiple Runs**: Run benchmarks multiple times
- **Hardware Specs**: Document test hardware
- **Environment**: Specify CUDA/PyTorch versions

### Example Benchmark Format

```python
# Performance benchmark example
import time
import torch

def benchmark_kernel():
    # Setup
    device = torch.device('cuda')
    size = 1024 * 1024
    
    # Warmup
    for _ in range(10):
        # Your kernel here
        pass
    
    # Benchmark
    start = time.time()
    for _ in range(100):
        # Your kernel here
        pass
    end = time.time()
    
    # Report
    avg_time = (end - start) / 100
    throughput = size / avg_time
    print(f"Average time: {avg_time:.6f}s")
    print(f"Throughput: {throughput:.2f} ops/s")
```

## 🐛 Bug Reports

### Reporting Issues

When reporting bugs, please include:

- **Hardware**: GPU model, driver version
- **Software**: CUDA version, PyTorch version
- **Steps**: Clear reproduction steps
- **Expected vs Actual**: What you expected vs what happened
- **Logs**: Error messages and logs

### Issue Template

```markdown
## Bug Description
Clear description of the issue

## Steps to Reproduce
1. Step 1
2. Step 2
3. Step 3

## Expected Behavior
What you expected to happen

## Actual Behavior
What actually happened

## Environment
- GPU: [Model]
- CUDA: [Version]
- PyTorch: [Version]
- OS: [Version]

## Additional Context
Any other relevant information
```

## 📚 Documentation Contributions

### README Updates

When updating documentation:

- **Clarity**: Make explanations clear and concise
- **Examples**: Include practical code examples
- **Links**: Add relevant links and references
- **Structure**: Maintain consistent formatting

### Code Comments

- **Purpose**: Explain what the code does
- **Parameters**: Document function parameters
- **Returns**: Document return values
- **Complexity**: Explain complex algorithms

## 🎯 Contribution Ideas

### High-Priority Areas

- **New CUDA Kernels**: Optimized implementations
- **PyTorch Optimizations**: Framework-specific improvements
- **Profiling Tools**: Better performance analysis
- **Architecture Support**: New GPU compatibility
- **Documentation**: Tutorials and guides

### Example Contributions

- **Memory Optimization**: New memory access patterns
- **Kernel Fusion**: Combining multiple operations
- **Tensor Core Usage**: Optimized matrix operations
- **Stream Management**: Better asynchronous execution
- **Distributed Training**: Multi-GPU optimizations

## 📞 Getting Help

### Community Resources

- **Issues**: Use GitHub issues for questions
- **Discussions**: Start discussions for ideas
- **Meetups**: Join our monthly meetups
- **YouTube**: Check our video tutorials

### Contact

- **GitHub Issues**: For bugs and feature requests
- **Discussions**: For questions and ideas
- **Email**: For private or sensitive matters

## 📄 License

By contributing to this project, you agree that your contributions will be licensed under the same license as the project (MIT License).

## 🙏 Recognition

Contributors will be recognized in:

- **README.md**: For significant contributions
- **Release Notes**: For each release
- **Documentation**: In relevant sections
- **Community**: In meetups and presentations

---

Thank you for contributing to the AI Performance Engineering community! 🚀
