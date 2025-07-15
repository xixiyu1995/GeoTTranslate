import os
import random
import re
import json
import pandas as pd
import torch
from datasets import Dataset
from modelscope import AutoTokenizer
from peft import PeftModel
from transformers import AutoModelForCausalLM
import numpy as np
from tqdm import tqdm
from functools import partial
from utils import dataset_jsonl_transfer, process_func, batch_predict, compute_metrics_test

# 全局实体标签
entity_labels = ['矿物', '岩石', '地层', '地质年代', '地质构造', '地名']

# 初始化 tokenizer
model_dir = 'Qwen3-1.7B'
tokenizer = AutoTokenizer.from_pretrained(
    model_dir,
    use_fast=False,
    trust_remote_code=True,
    padding_side='left'
)

# lora
model_train = r'.\output\GEO-Qwen3-old-7+8+32+10+1e-4+cosine+0.1\checkpoint-5410'
# 加载训练好的模型
print("Loading trained model...")
model = AutoModelForCausalLM.from_pretrained(model_train, device_map='auto', torch_dtype=torch.bfloat16)

# 使用新的数据集分割方式
# 总数据集路径
total_dataset_path = 'GNER_data/NERdata-10803.jsonl'
total_jsonl_new_path = 'GNER_data/NERdata-10803_total.jsonl'

# 转换总数据集格式（如果需要）
if not os.path.exists(total_jsonl_new_path):
    dataset_jsonl_transfer(total_dataset_path, total_jsonl_new_path)

# 加载总数据集
total_df = pd.read_json(total_jsonl_new_path, lines=True)

# 转换为 Dataset 对象
total_ds = Dataset.from_pandas(total_df)

# 按8:1:1比例划分训练集/验证集/测试集
split_ratio = total_ds.train_test_split(test_size=0.2, seed=42)  # 先分出20%
test_val = split_ratio['test'].train_test_split(test_size=0.5, seed=42)  # 20%再平分为验证集和测试集

# 创建最终数据集
# 我们只需要测试集
train_s = split_ratio['train']
val_ds = test_val['train']
test_ds = test_val['test']

# 打印测试集大小
print(f"测试集样本数: {len(test_ds)}")
assert len(test_ds) > 0, "测试集不能为空!"

# 读取测试集
test_df = test_ds.to_pandas().reset_index(drop=True)  # 取全部数据测试集
#test_df = test_ds.to_pandas().sample(n=50, random_state=42).reset_index(drop=True)  # 如果想快速测试，使用这行

# 进行预测并计算指标
test_text_list = []
pred_texts = []
ref_texts = []
input_texts = []  # 用于保存输入文本
batch_size = 8

for i in tqdm(range(0, len(test_df), batch_size), desc="Predicting"):
    try:
        batch_df = test_df.iloc[i:i + batch_size]
        messages_list = []
        batch_true_outputs = []

        for index, row in batch_df.iterrows():
            instruction = row['instruction']
            input_value = row['input']
            true_output = row['output']

            messages = [
                {'role': 'system', 'content': instruction},
                {'role': 'user', 'content': input_value}
            ]
            messages_list.append(messages)
            batch_true_outputs.append(true_output)

        responses = batch_predict(messages_list, model, tokenizer)

        for messages, response, true_output, input_value in zip(messages_list, responses, batch_true_outputs,
                                                                batch_df["input"]):
            messages.append({'role': 'assistant', 'content': response})
            result_text = f"{messages[0]}\n\n{messages[1]}\n\n{messages[2]}"
            test_text_list.append(result_text)  # 直接保存文本
            pred_texts.append(response)
            ref_texts.append(true_output)
            input_texts.append(input_value)  # 保存当前预测的输入文本

    except Exception as e:
        print(f"Error in batch {i // batch_size + 1}: {e}")
        continue

# 保存全部预测与参考为 Excel 文件
output_samples = []
for i, (input_text, pred, ref) in enumerate(zip(input_texts, pred_texts, ref_texts)):
    output_samples.append({
        '样本编号': f'Sample {i + 1}',
        '输入文本 (Input)': input_text,
        '预测结果 (Prediction)': pred,
        '参考答案 (Reference)': ref
    })

# 转换为 DataFrame
output_df = pd.DataFrame(output_samples)

# 保存为 Excel 文件
excel_output_path = 'prediction_results_all_1081_zh.xlsx'
output_df.to_excel(excel_output_path, index=False)
print(f"全部预测结果已保存至: {excel_output_path}")

# 计算指标
metrics = compute_metrics_test(pred_texts, ref_texts)
print("\n评估指标:")
print(f"精度 (Precision): {metrics['precision']:.4f}")
print(f"召回率 (Recall): {metrics['recall']:.4f}")
print(f"F1 值: {metrics['f1']:.4f}")

# 创建结果字典
result_dict = {
    'precision': metrics['precision'],
    'recall': metrics['recall'],
    'f1': metrics['f1']
}

# 直接打印结果字典
print("\n完整结果字典:")
print(json.dumps(result_dict, indent=2, ensure_ascii=False))

# 打印部分样本作为示例
print("\n前3个样本的预测结果:")
for i, text in enumerate(test_text_list[:3]):
    print(f"\n样本 {i+1}:")
    print(text)