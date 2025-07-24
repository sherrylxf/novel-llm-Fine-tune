from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer
from peft import PeftModel
import torch

# 路径配置
CHECKPOINT_DIR = "./model_output/checkpoint-145800"  # 最新checkpoint目录（你也可以用 final one）
TOKENIZER_DIR = "/home/fit/novel-llm/tokenizer"
FINAL_OUTPUT_DIR = "./final_model"

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, trust_remote_code=True)

# 加载最终模型（带LoRA结构）
model = AutoModelForCausalLM.from_pretrained(CHECKPOINT_DIR, trust_remote_code=True)
model = PeftModel.from_pretrained(model, CHECKPOINT_DIR)  # 加载LoRA adapter

# 保存
model.save_pretrained(FINAL_OUTPUT_DIR)
tokenizer.save_pretrained(FINAL_OUTPUT_DIR)

print("模型和 tokenizer 已保存至:", FINAL_OUTPUT_DIR)
