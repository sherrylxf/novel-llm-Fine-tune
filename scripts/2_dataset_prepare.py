import os
import re
import itertools
from datasets import Dataset
from transformers import AutoTokenizer

# ========== 配置 ==========
tokenizer_path = "./tokenizer"  # 已保存的 deepseek tokenizer
input_txt_path = "/home/fit/novel-llm/data/novels.txt"
output_dataset_path = "/home/fit/novel-llm/data/dataset"
block_size = 512  # 或者尝试 768、512

# ========== Step 1: 读取小说文本 ==========
with open(input_txt_path, "r", encoding="utf-8") as f:
    text = f.read()

# ========== Step 2: 智能分段 ==========
chapters = re.split(r"(第[\u4e00-\u9fa5]{1,10}章.*?)\n", text)
paragraphs = []

if len(chapters) > 1:
    for i in range(1, len(chapters), 2):
        title = chapters[i].strip()
        content = chapters[i + 1].strip() if i + 1 < len(chapters) else ""
        paragraph = f"{title}\n{content}"
        if len(paragraph) > 10:
            paragraphs.append(paragraph)
else:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

print(f"共提取段落/章节数：{len(paragraphs)}")

# ========== Step 3: 构建 Dataset ==========
dataset = Dataset.from_dict({"text": paragraphs})

# ========== Step 4: 加载 Tokenizer + 分词 ==========
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
    num_proc=1,
    batch_size=16,
    desc="Tokenizing"
)

# ========== Step 5: 拼接 tokens + 分块 ==========
def group_texts(examples):
    examples = {k: [x for x in examples[k] if len(x) > 0] for k in examples.keys()}
    concatenated = {k: list(itertools.chain.from_iterable(examples[k])) for k in examples.keys()}

    total_length = len(concatenated["input_ids"])
    if total_length < block_size:
        return {}

    total_length = (total_length // block_size) * block_size

    result = {
        k: [concatenated[k][i : i + block_size] for i in range(0, total_length, block_size)]
        for k in concatenated.keys()
    }
    result["labels"] = result["input_ids"].copy()
    return result

print("正在拼接并分块...")

lm_dataset = tokenized_dataset.map(
    group_texts,
    batched=True,
    batch_size=16,
    desc="分块中"
)

# ========== Step 6: 保存到硬盘 ==========
os.makedirs(output_dataset_path, exist_ok=True)
lm_dataset.save_to_disk(output_dataset_path)

print(f"处理完成，数据集已保存到：{output_dataset_path}")

