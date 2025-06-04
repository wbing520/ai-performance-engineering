import torch
import torch.quantization as quant
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).cuda().eval()

# Convert to INT8 (dynamic quantization for demonstration)
model_int8 = quant.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

# Inference
input_ids = tokenizer("Hello world", return_tensors="pt").input_ids.cuda()
with torch.no_grad():
    output = model_int8.generate(input_ids, max_length=50)
print(tokenizer.decode(output[0], skip_special_tokens=True))
