import random
import re  # 新增正则表达式模块
import json
import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from modelscope import AutoTokenizer
from swanlab.integration.transformers import SwanLabCallback
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import os
import swanlab
import numpy as np

# Global Entity Tags
#entity_labels = ['矿物', '岩石', '地层', '地质年代', '地质构造', '地名']
entity_labels = ['mineral', 'rock',  'stratum', 'geological time', 'ore deposit', 'location']


def compute_metrics(eval_preds, tokenizer, entity_labels):
    print("\n>>>> compute_metrics called <<<<")

    # Use the global tokenizer
    # global tokenizer

    # Initialize the regular expression pattern
    pattern = re.compile(
        r"{?['\"]?entity_text['\"]?\s*:\s*['\"]([^'\"]*)['\"]\s*,\s*['\"]?entity_label['\"]?\s*:\s*['\"]([^'\"]*)['\"]}?",
        re.IGNORECASE)

    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    # Decode the prediction results
    pred_ids = np.argmax(preds, axis=-1)
    pred_texts = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

    # Decode the true label
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    ref_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Initialize statistics variables
    total_correct = 0
    total_predicted = 0
    total_expected = 0

    for pred_str, ref_str in zip(pred_texts, ref_texts):
        # Parse the prediction results (using global entity_labels)
        pred_entities = set()
        for match in pattern.finditer(pred_str):
            text, label = match.groups()
            if label in entity_labels:
                pred_entities.add((text.strip(), label.strip()))

        # Parsing the real label
        ref_entities = set()
        for match in pattern.finditer(ref_str):
            text, label = match.groups()
            if label in entity_labels:
                ref_entities.add((text.strip(), label.strip()))

        if "No entity found" in pred_str:

            pred_entities = set()

        total_correct += len(pred_entities & ref_entities)
        total_predicted += len(pred_entities)
        total_expected += len(ref_entities)

    precision = total_correct / total_predicted if total_predicted > 0 else 0.0
    recall = total_correct / total_expected if total_expected > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0


    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4)
    }

def compute_metrics_test(pred_texts, ref_texts):
    print("\n>>>> compute_metrics called <<<<")
    pattern = re.compile(
        r"{?['\"]?entity_text['\"]?\s*:\s*['\"]([^'\"]*)['\"]\s*,\s*['\"]?entity_label['\"]?\s*:\s*['\"]([^'\"]*)['\"]}?",
        re.IGNORECASE)

    total_correct = 0
    total_predicted = 0
    total_expected = 0

    for pred_str, ref_str in zip(pred_texts, ref_texts):
        pred_entities = set()
        for match in pattern.finditer(pred_str):
            text, label = match.groups()
            if label in entity_labels:
                pred_entities.add((text.strip(), label.strip()))

        ref_entities = set()
        for match in pattern.finditer(ref_str):
            text, label = match.groups()
            if label in entity_labels:
                ref_entities.add((text.strip(), label.strip()))

        if "No entity found" in pred_str:
            pred_entities = set()

        total_correct += len(pred_entities & ref_entities)
        total_predicted += len(pred_entities)
        total_expected += len(ref_entities)

    precision = total_correct / total_predicted if total_predicted > 0 else 0.0
    recall = total_correct / total_expected if total_expected > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4)
    }


def dataset_jsonl_transfer(origin_path, new_path):
    '''
    Convert the original dataset into a new dataset in the data format required for fine-tuning the large model
    '''
    messages = []

    # Read old JSONL file
    with open(origin_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Parse each line of json data
            data = json.loads(line)
            input_text = data['text']
            entities = data['entities']
            match_names = ['mineral', 'rock',  'stratum', 'geological time', 'ore deposit', 'location']

            entity_sentence = ''
            for entity in entities:
                entity_json = dict(entity)
                entity_text = entity_json['entity_text']
                entity_names = entity_json['entity_names']
                for name in entity_names:
                    if name in match_names:
                        entity_label = name
                        break

                entity_sentence += f'''{{'entity_text': '{entity_text}', 'entity_label': '{entity_label}'}}'''

            if entity_sentence == '':
                entity_sentence = 'No entities found'

            message = {
                'instruction':'''You are an expert in text entity recognition. You need to extract 'mineral', 'rock',  'stratum', 'geological time', 'ore deposit', 'location'
                 entities from a given sentence. Output in json format, such as {'entity_text': 'Cambrian', 'entity_label': 'stratum'} 
                 Note: 1. Each line of output must be a correct json string. 
                 2. When no entity is found, output 'No entity found'. ''',

                'input': f'text:{input_text}',
                'output': entity_sentence,
            }

            messages.append(message)

    # Save the reconstructed JSONL file
    with open(new_path, 'w', encoding='utf-8') as file:
        for message in messages:
            file.write(json.dumps(message, ensure_ascii=False) + '\n')


def process_func(example, tokenizer):
    '''
    Preprocess the dataset
    '''
    MAX_LENGTH = 2048
    system_prompt = '''You are an expert in text entity recognition. You need to extract 'mineral', 'rock',  'stratum', 'geological time', 'ore deposit', 'location'
     entities from a given sentence. Output in json format, such as {'entity_text': 'Cambrian', 'entity_label': 'stratum'} 
    Note: 1. Each line of output must be a correct json string. 
    2. When no entity is found, output 'No entity found'. '''

    instruction = tokenizer(
        f'<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{example["input"]}<|im_end|>\n<|im_start|>assistant\n',
        add_special_tokens=False,
    )
    response = tokenizer(f'{example["output"]}', add_special_tokens=False)

    input_ids = instruction['input_ids'] + response['input_ids'] + [tokenizer.pad_token_id]
    attention_mask = instruction['attention_mask'] + response['attention_mask'] + [1]
    labels = [-100] * len(instruction['input_ids']) + response['input_ids'] + [tokenizer.pad_token_id]

    if len(input_ids) > MAX_LENGTH:  # Make a truncation
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}


def predict(messages, model, tokenizer):
    device = 'cuda'
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    model_inputs = tokenizer([text], return_tensors='pt').to(device)

    generated_ids = model.generate(
        #model_inputs.input_ids,
        **model_inputs,
        max_new_tokens=1024
    )
    # Extract only the part generated by the model (excluding the input part)
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
    response = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    print(response)
    return response

def batch_predict(messages_list, model, tokenizer):
    device = 'cuda'
    texts = [
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        ) for messages in messages_list
    ]
    model_inputs = tokenizer(
        texts,
        return_tensors='pt',
        padding='longest',
        truncation=True,
        add_special_tokens=False
    ).to(device)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    generated_ids = model.generate(
        model_inputs.input_ids,
        attention_mask=model_inputs.attention_mask,
        max_new_tokens=1024,
        pad_token_id=tokenizer.pad_token_id
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return responses
