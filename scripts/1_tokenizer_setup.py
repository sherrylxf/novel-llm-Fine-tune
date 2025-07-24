from transformers import AutoTokenizer

# 从已下载模型的目录加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained("D:/LLM/deepseek/deepseek-llm-7b-base")

# 保存副本到本地用于后续增量预训练
tokenizer.save_pretrained("./tokenizer/")
