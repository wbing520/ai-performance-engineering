// Architecture-specific optimizations for CUDA 12.9
// Supports Hopper H100/H200 (sm_90) and Blackwell B200/B300 (sm_100)
// nvtx_profiling.cu
// NVTX annotation examples for profiling inference workloads

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <stdio.h>
#include <vector>
#include <chrono>

// Simulated model components
struct Token {
    int id;
    float embedding[512];
};

class SimpleModel {
private:
    float* d_weights;
    float* d_cache;
    float* d_temp;
    int model_dim;
    int cache_size;

public:
    SimpleModel(int dim = 4096, int cache_sz = 1024*1024) : model_dim(dim), cache_size(cache_sz) {
        cudaMalloc(&d_weights, model_dim * model_dim * sizeof(float));
        cudaMalloc(&d_cache, cache_size * sizeof(float));
        cudaMalloc(&d_temp, model_dim * sizeof(float));
        
        // Initialize with dummy data
        cudaMemset(d_weights, 0, model_dim * model_dim * sizeof(float));
        cudaMemset(d_cache, 0, cache_size * sizeof(float));
    }

    ~SimpleModel() {
        cudaFree(d_weights);
        cudaFree(d_cache);
        cudaFree(d_temp);
    }

    void encode(const std::vector<Token>& prompt_tokens);
    Token decode_next();
    void attention_computation(int seq_len);
    void feedforward_computation();
    void update_kv_cache(int position);
};

// Kernels for simulation
__global__ void attention_kernel(float* weights, float* cache, float* output, 
                                int seq_len, int model_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < seq_len * model_dim) {
        float sum = 0.0f;
        for (int i = 0; i < model_dim; ++i) {
            sum += weights[idx * model_dim + i] * cache[i];
        }
        output[idx] = tanhf(sum * 0.001f);
    }
}

__global__ void feedforward_kernel(float* input, float* weights, float* output, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dim) {
        float sum = 0.0f;
        for (int i = 0; i < dim; ++i) {
            sum += input[i] * weights[idx * dim + i];
        }
        output[idx] = fmaxf(0.0f, sum); // ReLU activation
    }
}

__global__ void cache_update_kernel(float* cache, float* new_data, int position, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dim) {
        cache[position * dim + idx] = new_data[idx];
    }
}

void SimpleModel::encode(const std::vector<Token>& prompt_tokens) {
    // NVTX range for the entire encode operation
    nvtxRangePush("Model::encode");
    
    int seq_len = prompt_tokens.size();
    printf("Encoding %d tokens...\n", seq_len);
    
    // Simulate processing each token
    for (int i = 0; i < seq_len; ++i) {
        // NVTX range for each token processing
        char token_label[64];
        snprintf(token_label, sizeof(token_label), "Token_%d", i);
        nvtxRangePush(token_label);
        
        // Attention computation
        attention_computation(i + 1);
        
        // Feed-forward computation  
        feedforward_computation();
        
        // Update KV cache
        update_kv_cache(i);
        
        nvtxRangePop(); // End token processing
    }
    
    nvtxRangePop(); // End encode
}

Token SimpleModel::decode_next() {
    // NVTX range for decode step
    nvtxRangePush("Model::decode_next");
    
    // Attention computation
    attention_computation(1);
    
    // Feed-forward computation
    feedforward_computation();
    
    // Generate next token (simplified)
    Token next_token;
    next_token.id = rand() % 1000;
    
    nvtxRangePop(); // End decode
    return next_token;
}

void SimpleModel::attention_computation(int seq_len) {
    // NVTX range with custom color
    nvtxEventAttributes_t eventAttrib = {0};
    eventAttrib.version = NVTX_VERSION;
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.colorType = NVTX_COLOR_ARGB;
    eventAttrib.color = 0xFF00FF00; // Green
    eventAttrib.messageType = NVTX_MESSAGE_STRING;
    eventAttrib.message.ascii = "Attention";
    nvtxRangePushEx(&eventAttrib);
    
    dim3 block(256);
    dim3 grid((seq_len * model_dim + block.x - 1) / block.x);
    
    attention_kernel<<<grid, block>>>(d_weights, d_cache, d_temp, seq_len, model_dim);
    cudaDeviceSynchronize();
    
    nvtxRangePop();
}

void SimpleModel::feedforward_computation() {
    // NVTX range with blue color
    nvtxEventAttributes_t eventAttrib = {0};
    eventAttrib.version = NVTX_VERSION;
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.colorType = NVTX_COLOR_ARGB;
    eventAttrib.color = 0xFF0000FF; // Blue
    eventAttrib.messageType = NVTX_MESSAGE_STRING;
    eventAttrib.message.ascii = "FeedForward";
    nvtxRangePushEx(&eventAttrib);
    
    dim3 block(256);
    dim3 grid((model_dim + block.x - 1) / block.x);
    
    feedforward_kernel<<<grid, block>>>(d_temp, d_weights, d_temp, model_dim);
    cudaDeviceSynchronize();
    
    nvtxRangePop();
}

void SimpleModel::update_kv_cache(int position) {
    // NVTX range with yellow color
    nvtxEventAttributes_t eventAttrib = {0};
    eventAttrib.version = NVTX_VERSION;
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.colorType = NVTX_COLOR_ARGB;
    eventAttrib.color = 0xFFFFFF00; // Yellow
    eventAttrib.messageType = NVTX_MESSAGE_STRING;
    eventAttrib.message.ascii = "KV_Cache_Update";
    nvtxRangePushEx(&eventAttrib);
    
    dim3 block(256);
    dim3 grid((model_dim + block.x - 1) / block.x);
    
    cache_update_kernel<<<grid, block>>>(d_cache, d_temp, position, model_dim);
    cudaDeviceSynchronize();
    
    nvtxRangePop();
}

// Simulated inference pipeline with comprehensive NVTX annotations
void run_inference_with_profiling(const std::vector<Token>& prompt_tokens, 
                                  SimpleModel& model, int num_generate_steps) {
    
    // Mark the entire inference session
    nvtxRangePush("Inference_Session");
    
    printf("=== Starting Inference Session ===\n");
    printf("Prompt tokens: %zu\n", prompt_tokens.size());
    printf("Generation steps: %d\n", num_generate_steps);
    
    // Phase 1: Prefill stage
    {
        nvtxEventAttributes_t eventAttrib = {0};
        eventAttrib.version = NVTX_VERSION;
        eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        eventAttrib.colorType = NVTX_COLOR_ARGB;
        eventAttrib.color = 0xFFFF0000; // Red for prefill
        eventAttrib.messageType = NVTX_MESSAGE_STRING;
        eventAttrib.message.ascii = "Prefill_Stage";
        nvtxRangePushEx(&eventAttrib);
        
        auto start = std::chrono::high_resolution_clock::now();
        model.encode(prompt_tokens);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        printf("Prefill completed in %ld ms\n", duration.count());
        
        nvtxRangePop(); // End prefill
    }
    
    // Phase 2: Decode stage - generate tokens one by one
    {
        nvtxEventAttributes_t eventAttrib = {0};
        eventAttrib.version = NVTX_VERSION;
        eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        eventAttrib.colorType = NVTX_COLOR_ARGB;
        eventAttrib.color = 0xFF800080; // Purple for decode
        eventAttrib.messageType = NVTX_MESSAGE_STRING;
        eventAttrib.message.ascii = "Decode_Stage";
        nvtxRangePushEx(&eventAttrib);
        
        auto decode_start = std::chrono::high_resolution_clock::now();
        
        std::vector<Token> generated_tokens;
        for (int t = 0; t < num_generate_steps; ++t) {
            // Mark each decode step individually
            char step_label[64];
            snprintf(step_label, sizeof(step_label), "Decode_Step_%d", t);
            nvtxRangePush(step_label);
            
            Token next_token = model.decode_next();
            generated_tokens.push_back(next_token);
            
            nvtxRangePop(); // End decode step
        }
        
        auto decode_end = std::chrono::high_resolution_clock::now();
        auto decode_duration = std::chrono::duration_cast<std::chrono::milliseconds>(decode_end - decode_start);
        
        printf("Generated %d tokens in %ld ms\n", num_generate_steps, decode_duration.count());
        printf("Average time per token: %.2f ms\n", 
               (float)decode_duration.count() / num_generate_steps);
        
        nvtxRangePop(); // End decode stage
    }
    
    nvtxRangePop(); // End inference session
    
    printf("=== Inference Session Complete ===\n");
}

// Batch processing with NVTX annotations
void run_batch_inference(const std::vector<std::vector<Token>>& batch_prompts,
                        SimpleModel& model, int num_generate_steps) {
    
    nvtxRangePush("Batch_Inference");
    
    printf("=== Batch Inference ===\n");
    printf("Batch size: %zu\n", batch_prompts.size());
    
    for (size_t i = 0; i < batch_prompts.size(); ++i) {
        char batch_label[64];
        snprintf(batch_label, sizeof(batch_label), "Request_%zu", i);
        nvtxRangePush(batch_label);
        
        run_inference_with_profiling(batch_prompts[i], model, num_generate_steps);
        
        nvtxRangePop();
    }
    
    nvtxRangePop(); // End batch inference
}

// Communication simulation with NVTX
void simulate_multi_gpu_communication() {
    nvtxRangePush("Multi_GPU_Communication");
    
    const int data_size = 1024 * 1024; // 1M floats
    float *d_data1, *d_data2;
    
    cudaMalloc(&d_data1, data_size * sizeof(float));
    cudaMalloc(&d_data2, data_size * sizeof(float));
    
    // Simulate all-reduce operation
    {
        nvtxEventAttributes_t eventAttrib = {0};
        eventAttrib.version = NVTX_VERSION;
        eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        eventAttrib.colorType = NVTX_COLOR_ARGB;
        eventAttrib.color = 0xFFFFA500; // Orange
        eventAttrib.messageType = NVTX_MESSAGE_STRING;
        eventAttrib.message.ascii = "AllReduce_Communication";
        nvtxRangePushEx(&eventAttrib);
        
        // Simulate P2P copy
        cudaMemcpy(d_data2, d_data1, data_size * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaDeviceSynchronize();
        
        nvtxRangePop();
    }
    
    // Simulate all-to-all operation (for MoE)
    {
        nvtxEventAttributes_t eventAttrib = {0};
        eventAttrib.version = NVTX_VERSION;
        eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        eventAttrib.colorType = NVTX_COLOR_ARGB;
        eventAttrib.color = 0xFF00FFFF; // Cyan
        eventAttrib.messageType = NVTX_MESSAGE_STRING;
        eventAttrib.message.ascii = "AllToAll_Communication";
        nvtxRangePushEx(&eventAttrib);
        
        // Simulate multiple smaller transfers
        for (int i = 0; i < 4; ++i) {
            char transfer_label[64];
            snprintf(transfer_label, sizeof(transfer_label), "Transfer_%d", i);
            nvtxRangePush(transfer_label);
            
            int chunk_size = data_size / 4;
            cudaMemcpy(d_data2 + i * chunk_size, d_data1 + i * chunk_size, 
                      chunk_size * sizeof(float), cudaMemcpyDeviceToDevice);
            
            nvtxRangePop();
        }
        
        cudaDeviceSynchronize();
        nvtxRangePop();
    }
    
    cudaFree(d_data1);
    cudaFree(d_data2);
    nvtxRangePop(); // End multi-GPU communication
}

int main() {
    printf("NVTX Profiling Examples for Inference - Chapter 16\n");
    printf("==================================================\n");
    
    // Initialize NVTX
    nvtxInitialize(NULL);
    
    // Create model
    SimpleModel model(2048, 512*1024); // Smaller model for demo
    
    // Create sample prompt
    std::vector<Token> prompt_tokens;
    for (int i = 0; i < 50; ++i) {
        Token token;
        token.id = i;
        prompt_tokens.push_back(token);
    }
    
    printf("\n=== Single Request Inference ===\n");
    run_inference_with_profiling(prompt_tokens, model, 10);
    
    printf("\n=== Batch Inference ===\n");
    std::vector<std::vector<Token>> batch_prompts;
    for (int b = 0; b < 3; ++b) {
        std::vector<Token> batch_prompt;
        for (int i = 0; i < 20 + b * 10; ++i) { // Variable prompt lengths
            Token token;
            token.id = i + b * 100;
            batch_prompt.push_back(token);
        }
        batch_prompts.push_back(batch_prompt);
    }
    
    run_batch_inference(batch_prompts, model, 5);
    
    printf("\n=== Multi-GPU Communication Simulation ===\n");
    simulate_multi_gpu_communication();
    
    printf("\n=== Profiling Summary ===\n");
    printf("NVTX annotations have been added throughout the inference pipeline.\n");
    printf("Use the following commands to capture and analyze the profile:\n\n");
    
    printf("1. Capture profile with Nsight Systems:\n");
    printf("   nsys profile --force-overwrite=true -o inference_profile ./nvtx_profiling\n\n");
    
    printf("2. Open the profile in Nsight Systems GUI:\n");
    printf("   nsight-sys inference_profile.nsys-rep\n\n");
    
    printf("3. Profile specific kernels with Nsight Compute:\n");
    printf("   ncu --section SpeedOfLight --section MemoryWorkloadAnalysis ./nvtx_profiling\n\n");
    
    printf("Key NVTX regions to look for in the timeline:\n");
    printf("- Inference_Session (overall session)\n");
    printf("- Prefill_Stage (red) - prompt processing\n");
    printf("- Decode_Stage (purple) - token generation\n");
    printf("- Attention (green) - attention computations\n");
    printf("- FeedForward (blue) - MLP layers\n");
    printf("- KV_Cache_Update (yellow) - cache operations\n");
    printf("- AllReduce_Communication (orange) - collective ops\n");
    printf("- AllToAll_Communication (cyan) - MoE routing\n\n");
    
    printf("Analysis tips:\n");
    printf("- Look for GPU idle gaps between operations\n");
    printf("- Check for overlapping computation and communication\n");
    printf("- Identify bottlenecks in prefill vs decode phases\n");
    printf("- Monitor memory transfers and cache updates\n");
    printf("- Verify proper kernel launch patterns\n");
    
    return 0;
}

// CUDA 12.9 Stream-ordered Memory Allocation Example
__global__ void stream_ordered_memory_example() {
    // Example of stream-ordered memory allocation
    // This is a placeholder for actual implementation
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Your kernel code here
}

// CUDA 12.9 TMA (Tensor Memory Accelerator) Example
__global__ void tma_example() {
    // Example of TMA usage for Blackwell B200/B300
    // This is a placeholder for actual implementation
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Your TMA code here
}
