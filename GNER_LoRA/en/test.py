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
from en_utils import dataset_jsonl_transfer, process_func, batch_predict, compute_metrics_test

# Global Entity Tags
#entity_labels = ['矿物', '岩石', '地层', '地质年代', '地质构造', '地名']
entity_labels = ['mineral', 'rock',  'stratum', 'geological time', 'ore deposit', 'location']

# Initialize tokenizer and model
model_dir = 'Qwen3-1.7B'
tokenizer = AutoTokenizer.from_pretrained(
    model_dir,
    use_fast=False,
    trust_remote_code=True,
    padding_side='left'
)

# lora
model_train = r'.\output\GEO-Qwen3-7+8+32+5+1e-4+cosine+0.1\checkpoint-7990'
print("Loading trained model...")
model = AutoModelForCausalLM.from_pretrained(model_train, device_map='auto', torch_dtype=torch.bfloat16)


total_dataset_path = 'GNER_data/OzROCK_all.jsonl'
total_jsonl_new_path = 'GNER_data/OzROCK_total.jsonl'

if not os.path.exists(total_jsonl_new_path):
    dataset_jsonl_transfer(total_dataset_path, total_jsonl_new_path)

total_df = pd.read_json(total_jsonl_new_path, lines=True)

total_ds = Dataset.from_pandas(total_df)

# Divide the training set/validation set/test set into a ratio of 8:1:1
split_ratio = total_ds.train_test_split(test_size=0.2, seed=42)
test_val = split_ratio['test'].train_test_split(test_size=0.5, seed=42)

# Create the final dataset
# We only need the test set
train_s = split_ratio['train']
val_ds = test_val['train']
test_ds = test_val['test']

# Print the size of the test set
print(f"Number of test set samples: {len(test_ds)}")
assert len(test_ds) > 0, "The test set cannot be empty!"

# Read the test set
test_df = test_ds.to_pandas().reset_index(drop=True) # Get the entire data test set
#test_df = test_ds.to_pandas().sample(n=100, random_state=42).reset_index(drop=True) # If you want to test quickly, use this line

# Make predictions and calculate indicators
test_text_list = []
pred_texts = []
ref_texts = []
input_texts = [] # Used to save input text
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
            test_text_list.append(result_text)
            pred_texts.append(response)
            ref_texts.append(true_output)
            input_texts.append(input_value)

    except Exception as e:
        print(f"Error in batch {i // batch_size + 1}: {e}")
        continue

# Save all predictions and references as Excel files
output_samples = []
for i, (input_text, pred, ref) in enumerate(zip(input_texts, pred_texts, ref_texts)):
    output_samples.append({
        'Sample number': f'Sample {i + 1}',
        'Input text (Input)': input_text,
        'Prediction result (Prediction)': pred,
        'Reference answer (Reference)': ref
    })

output_df = pd.DataFrame(output_samples)

# Save as Excel file
"""excel_output_path = 'prediction_results_all_3195.xlsx'
output_df.to_excel(excel_output_path, index=False)
print(f"All prediction results have been saved to: {excel_output_path}")
"""

# Calculate metrics
metrics = compute_metrics_test(pred_texts, ref_texts)
print("\nEvaluation metrics:")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
print(f"F1 value: {metrics['f1']:.4f}")

# Create result dictionary
result_dict = {
    'precision': metrics['precision'],
    'recall': metrics['recall'],
    'f1': metrics['f1']
}

# Print the result dictionary directly
print("\nComplete result dictionary:")
print(json.dumps(result_dict, indent=2, ensure_ascii=False))

# Print some samples as examples
print("\nPrediction results of the first 3 samples:")
for i, text in enumerate(test_text_list[:3]):
    print(f"\nSample {i+1}:")
    print(text)