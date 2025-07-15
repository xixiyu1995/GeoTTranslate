import json


def convert_jsonl_to_format(input_file, output_file):
    # 读取JSONL文件并转换为目标格式
    json_data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            try:
                data = json.loads(line)
                labels = []

                # 转换实体格式
                for i, entity in enumerate(data.get("entities", [])):
                    # 生成唯一实体ID (T0, T1, T2...)
                    entity_id = f"T{i}"

                    # 注意：原始数据的end_idx是包含式索引，转换为不包含式结束位置
                    start_idx = entity["start_idx"]
                    end_idx = entity["end_idx"] + 1  # 转换为不包含式结束索引
                    entity_text = entity["entity_text"]
                    entity_label = entity["entity_label"]

                    labels.append([entity_id, entity_label, start_idx, end_idx, entity_text])

                # 添加转换后的数据
                json_data.append({
                    "text": data["text"],
                    "labels": labels
                })

            except json.JSONDecodeError:
                print(f"Error parsing line {idx}: {line}")

    # 保存为JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    print(f"Converted {len(json_data)} records to {output_file}")


# 使用示例
if __name__ == "__main__":
    input_file = "NERdata_open_all_zh.jsonl"  # 替换为您的输入文件路径
    output_file = "NERdata_all_bert.jsonl"  # 输出文件路径
    convert_jsonl_to_format(input_file, output_file)


