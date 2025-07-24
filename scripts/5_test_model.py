from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# 原始模型加载（未微调）
tokenizer = AutoTokenizer.from_pretrained("/home/fit/novel-llm/tokenizer", trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    "/home/fit/novel-llm/model/deepseek-llm-7b-base",
    device_map="auto",
    trust_remote_code=True
)

base_pipe = pipeline("text-generation", model=base_model, tokenizer=tokenizer)

prompt = "李青云从昏迷中苏醒，发现自己身处一座陌生的山洞中，体内灵力微弱。他记得自己在与魔修对战中身受重伤……接下来他会？"
output_base = base_pipe(prompt, max_new_tokens=100, do_sample=True, temperature=0.9)[0]['generated_text']
print("原始模型输出：\n", output_base)


from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained("./final_model", trust_remote_code=True)

# 加载 base 模型 + LoRA adapter
base_model = AutoModelForCausalLM.from_pretrained(
    "/home/fit/novel-llm/model/deepseek-llm-7b-base",
    trust_remote_code=True,
    device_map="auto",  # accelerate 启用，会自动分配到 GPU
)
model = PeftModel.from_pretrained(base_model, "./final_model")

# 不加 device=0（避免冲突）
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# 推理测试
prompt = "李青云从昏迷中苏醒，发现自己身处一座陌生的山洞中，体内灵力微弱。他记得自己在与魔修对战中身受重伤……接下来他会？"
output = pipe(prompt, max_new_tokens=100, do_sample=True, temperature=0.9)
print("微调后模型生成结果：\n", output[0]["generated_text"])
