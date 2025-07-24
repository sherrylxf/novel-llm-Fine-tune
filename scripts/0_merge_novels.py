import re
import os

def clean_novel_text(text):
    """
    清洗小说文本，统一章节格式，规范标点、空行等格式。
    """
    # 1. 去除广告和无用内容
    patterns_to_remove = [
        r'(www\.\w+\.com|请访问|小说来自|更多精彩小说|最新章节请登录)',  # 广告等内容
    ]
    for pattern in patterns_to_remove:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    # 2. 替换非标准中文标点和空格
    replacements = {
        '　': ' ',  # 全角空格
        '【': '「',
        '】': '」',
        '“': '「',
        '”': '」',
        '‘': '『',
        '’': '』',
        '"': '」',  # 英文引号转中文
        "'": '』',
        '\u3000': ' ',  # 全角空格
    }
    for key, val in replacements.items():
        text = text.replace(key, val)

    # 3. 标准化章节标题（“第X章 名称”）
    def standardize_chapter_title(match):
        number = match.group(1)
        title = match.group(2).strip()
        return f'\n\n第{number}章 {title}\n'

    chapter_pattern = r'(第[\d一二三四五六七八九十百千零〇两]+章)[\s·:-—]*(.*)'
    text = re.sub(chapter_pattern, lambda m: f'\n\n{m.group(1)} {m.group(2).strip()}\n', text)

    # 4. 统一段落空行（3行以上合并成2行）
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 5. 移除段首多余空格
    text = re.sub(r'^[ \t]+', '', text, flags=re.MULTILINE)

    # 6. 移除乱码和杂项字符（中文、英文标点外的字符）
    allowed_chars = r'\x00-\xFF\u4e00-\u9fa5，。！？：“”‘’《》、；（）【】…—\n\s'
    text = re.sub(f'[^{allowed_chars}]', '', text)

    # 7. 去除首尾空行
    text = text.strip()

    return text


# 完整的文件合并与处理
input_folder = "data/novels_raw"
output_file = "data/novels.txt"

with open(output_file, 'w', encoding='utf-8') as outfile:
    for filename in sorted(os.listdir(input_folder)):
        if filename.endswith(".txt"):
            filepath = os.path.join(input_folder, filename)
            with open(filepath, 'r', encoding='utf-8') as infile:
                content = infile.read()
                cleaned_content = clean_novel_text(content)
                outfile.write(cleaned_content + "\n\n")
print(f"合并完成，写入到 {output_file}")