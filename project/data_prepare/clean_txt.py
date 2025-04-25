import re

input_file = r"C:/Users/Acer/Desktop/cs4486/result/all_names.txt"
output_file = r"C:/Users/Acer/Desktop/cs4486/result/cleaned_names.txt"

pattern_timestamp = re.compile(r'_\d{8}_\d{6}')  # 匹配时间戳格式 _YYYYMMDD_HHMMSS

with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
    for line in f_in:
        line = line.strip()
        if not line:
            continue

        # 把反斜杠替换为正斜杠
        line = line.replace('\\', '/')

        # 取文件名（含扩展名）
        filename = line.split('/')[-1]

        # 去除时间戳
        cleaned_name = pattern_timestamp.sub('', filename)

        f_out.write(cleaned_name + '\n')

print(f"已处理完毕，结果输出到 {output_file}")