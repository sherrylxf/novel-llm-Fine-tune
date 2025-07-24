# novel-llm-Fine-tune 基于修仙小说的语言模型微调项目

本项目旨在基于开源语言模型 Deepseek 对修仙类小说进行微调，生成贴合网络小说风格的文本。项目支持自定义小说语料、分词器训练、高频词提取、模型保存与测试。

---

## 项目结构

```bash
novel-llm/
│
├── data/                   # 存放原始小说和中间数据
│   ├── novels_raw/         # 未清洗处理的小说
│   ├── novels.txt          # 清洗处理后的合并小说
│   ├── dataset/            # 合并小说的数据集
├── tokenizer/              # 自定义 tokenizer 输出（可选）
├── model/                  # 原始预训练模型
├── final_model/            # 微调后保存的模型和 adapter
├── model_output/           # 生成文本输出
├── scripts/                # 项目脚本目录
│   ├── 0_merge_novels.py         # 合并小说语料
│   ├── 1_tokenizer_setup.py      # 构建自定义 tokenizer（可选）
│   ├── 2_dataset_prepare.py      # 构造训练数据集
│   ├── 3_trainer_run.py          # 执行模型微调训练
│   ├── 4_save_model.py           # 保存微调模型和 adapter
│   ├── 5_test_model.py           # 测试生成效果
```

---

## 环境安装

建议使用 Anaconda 虚拟环境。

```bash
conda create -n novel-llm python=3.10
conda activate novel-llm

# 安装依赖
pip install -r requirements.txt
```

## 下载模型

本项目使用 DeepSeek 基础模型（7B）。

```bash
# 使用 huggingface-cli 下载
huggingface-cli login
git lfs install
git clone https://huggingface.co/deepseek-ai/deepseek-llm-7b-base model/
```

---

## 训练流程

按顺序执行以下脚本：

```bash
# 步骤 0：合并小说语料
python scripts/0_merge_novels.py

# 步骤 1：训练自定义 tokenizer（可选）
python scripts/1_tokenizer_setup.py

# 步骤 2：构造训练数据集
python scripts/2_dataset_prepare.py

# 步骤 3：进行微调训练
python scripts/3_trainer_run.py

# 步骤 4：保存模型和 adapter
python scripts/4_save_model.py

# 步骤 5：测试生成效果
python scripts/5_test_model.py
```

---

## 测试 Prompt 示例

你可以在 `5_test_model.py` 中替换 `prompt` 字符串，测试不同的情境：

```python
prompt = "李青云从昏迷中醒来，发现自己体内的灵力几乎耗尽，四周是一片荒凉的山谷。"
```


---

## 微调模型目录（`final_model/`）

| 文件名                         | 说明          |
| --------------------------- | ----------- |
| `adapter_model.safetensors` | LoRA 微调后的权重 |
| `adapter_config.json`       | LoRA 配置文件   |
| `tokenizer.json`            | 自定义分词器模型    |
| `tokenizer_config.json`     | 分词器配置       |
| `special_tokens_map.json`   | 特殊 token 映射 |
| `README.md`                 | 模型说明文档（可选）  |

---

## 常见问题

### 1. 报错：`ValueError: The model has been loaded with accelerate...`

**解决方法**：不要传 `device=0` 到 `pipeline()` 中，修改如下：

```python
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)  # 去掉 device 参数
```

### 2. 生成很慢 / CPU 被强制使用？

* 原因可能是 GPU 显存不足，导致 offload 到 CPU。
* 解决方案：

  * 减少 batch size；
  * 降低 max length；
  * 加 `offload_buffers=True`；
  * 使用更小模型（如 DeepSeek-Coder-1.3B）。

---

## 示例输出对比

| Prompt      | 原始模型输出   | 微调后模型输出     |
| ----------- | -------- | ----------- |
| 李青云从昏迷中醒来…… | 多为通用场景描述 | 更贴合修仙小说语言风格 |

---

