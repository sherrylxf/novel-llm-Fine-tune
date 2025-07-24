import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ------------ 显存优化设置 ------------
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cuda.matmul.allow_tf32 = True  # 启用TF32加速
torch.cuda.empty_cache()

# ------------ 设备检测 ------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("当前设备:", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("可用显存:", torch.cuda.get_device_properties(0).total_memory / 1024**3, "GB")
else:
    print("未使用GPU，训练速度将非常慢！")

# ------------ 路径配置 ------------
TOKENIZER_DIR = "/home/fit/novel-llm/tokenizer"
MODEL_DIR = "/home/fit/novel-llm/model/deepseek-llm-7b-base"
DATASET_DIR = "/home/fit/novel-llm/data/dataset"

# ------------ 4bit量化配置 ------------
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# ------------ 加载tokenizer ------------
print("正在加载tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # 设置pad token

# ------------ 加载模型 ------------
print("正在加载模型（4bit量化 + 准备kbit训练）...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    trust_remote_code=True,
    quantization_config=quant_config,
    device_map="auto",
    low_cpu_mem_usage=True
)

# 准备模型以支持4bit训练（LoRA需要）
model = prepare_model_for_kbit_training(model)

# ------------ 配置LoRA ------------
print("配置LoRA适配器...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 显存优化
model.gradient_checkpointing_enable()
model.config.use_cache = False  # 关闭缓存，节省显存

# ------------ 加载数据集 ------------
print("正在加载数据集...")
dataset = load_from_disk(DATASET_DIR)

print("数据集示例:", dataset[0])  # 确认dataset已经包含input_ids等字段

# ------------ 数据收集器 ------------
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ------------ 训练参数配置 ------------
training_args = TrainingArguments(
    output_dir="./model_output",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-4,
    optim="paged_adamw_8bit",
    logging_steps=10,
    save_steps=200,
    save_total_limit=2,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    report_to="none",
    remove_unused_columns=False,  # 重要！保持所有数据列
    fp16=False,
    bf16=False,
    dataloader_num_workers=2,
    gradient_checkpointing=True
)

# ------------ 创建 Trainer ------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

# ------------ 启动训练（从 checkpoint-31000 继续）------------
print("开始训练（断点续训）...")
trainer.train(resume_from_checkpoint="./model_output/checkpoint-xxxxxx") # xxxxxx为最后训练的 checkpoint 文件
print("训练完成！")

