from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", padding_side="left")

quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", quantization_config=quantization_config, device_map="auto", low_cpu_mem_usage=True)

input_text = "What is the best way to spend Holy week being a Catholic according to the Church fathers?"
model_inputs = tokenizer([input_text], return_tensors="pt").to("cuda")

generated_text = model.generate(**model_inputs, max_length=1000, pad_token_id=tokenizer.eos_token_id)

result = tokenizer.batch_decode(generated_text, skip_special_tokens=True)[0]
print(result)