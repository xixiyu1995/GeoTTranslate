import os
import json
import re
import numpy as np
import pandas as pd
import torch
import gradio as gr
from pymilvus import connections, Collection
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

bert_model_path = "BERT"
bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_path)
bert_model = AutoModel.from_pretrained(bert_model_path)
bert_model.to(device)

tokenizer = AutoTokenizer.from_pretrained('Qwen3-1.7B', use_fast=False,
                                          trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained('Qwen3-1.7B', device_map="auto",
                                             torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model,
                                  model_id="checkpoint")
model.to(device)

deepseek_model_path = "DeepSeek-R1-Distill-Qwen-7B"
deepseek_tokenizer = AutoTokenizer.from_pretrained(deepseek_model_path)
deepseek_model = AutoModelForCausalLM.from_pretrained(deepseek_model_path, torch_dtype=torch.float16, device_map="auto")

connections.connect("default", uri="http://127.0.0.1:19530")
collection_name = "Dictdata_collection"
collection = Collection(name=collection_name)


def get_vector(text):
    """ 使用 BERT 将输入文本转换为向量;Convert input text into vectors using BERT """
    text = text.lower()
    inputs = bert_tokenizer(text, return_tensors='pt').to(device)
    outputs = bert_model(**inputs)
    vector = outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy()
    return vector.astype(np.float32)


def predict(messages, model, tokenizer):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in
                     zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response


def search_in_milvus(collection, query_vector, top_k=3):
    """
    Search for the most similar translation words in Milvus
    :param collection: Milvus collection
    :param query_vector: query vector
    :param top_k: number of most similar items returned
    :return: (translation candidate, similarity)
    在 Milvus 中查询最相似的翻译词
    :param collection: Milvus 集合
    :param query_vector: 查询向量
    :param top_k: 返回的最相似项数
    :return: (翻译候选词, 相似度)
    """
    query_vector = query_vector.flatten()
    search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
    results = collection.search(
        data=[query_vector],
        anns_field="english_vector",
        param=search_params,
        limit=top_k,
        output_fields=["chinese_word"]
    )

    if results and len(results[0]) > 0:
        return [(hit.entity.get("chinese_word"), hit.distance) for hit in results[0]]

    return []


def analyze_text(input_text):
    """ Identify geological entities and return annotated text;识别地质实体，并返回标注后的文本 """
    instruction = '''You are an expert in text entity recognition. You need to extract 'mineral', 'rock',  'stratum', 'geological time', 'ore deposit', 'location'
                 entities from a given sentence. Output in json format, such as {'entity_text': 'Cambrian', 'entity_label': 'stratum'} 
                 Note: 1. Each line of output must be a correct json string. 
                 2. When no entity is found, output 'No entity found'. '''

    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": f"text:{input_text}"}
    ]

    response = predict(messages, model, tokenizer)

    try:
        if response.startswith("[") and response.endswith("]"):
            entities = json.loads(response)
        else:
            json_objects = re.findall(r"\{.*?\}", response)
            json_fixed = "[" + ",".join(json_objects) + "]"
            json_fixed = json_fixed.replace("'", '"')
            entities = json.loads(json_fixed)

        if not isinstance(entities, list):
            raise ValueError("output format error")


        if not entities:
            return input_text, []

        return input_text, [(ent["entity_text"], ent["entity_label"]) for ent in entities]

    except Exception as e:
        print(f"JSON parsing error: {e}, response: {response}")
        return input_text, []


def dynamic_translate(input_text, entity_candidates):
    glossary_notes = ""
    if entity_candidates:
        glossary_notes += "\n\nGeological terminology glossary (please prioritize these translations if appropriate):\n"

        for entity_text, entity_label, candidates in entity_candidates:
            if candidates:
                glossary_notes += f"- {entity_text} ({entity_label}): {candidates[0]}\n"

    #  system prompt
    system_prompt = f"""你是一位专业的翻译人员，专注于地质领域文献翻译。你的任务是将英文地质文本**准确、科学地翻译成中文**，请遵循以下原则：

    1. 最大程度保持原句结构和语法逻辑；
    2. 精准传达专业术语和地质学概念；
    3. 特别关注术语表中列出的**地质实体类别**（如岩石、矿物、地层等）；
    4. 在适当的情况下优先使用建议术语，确保翻译与语境和实体类型相符；
    5. 不添加解释或评论，只输出**纯粹的中文翻译结果**；
    6. 翻译风格需正式、专业，符合学术或技术出版标准。

    术语建议列表（按实体类别分类）如下：
    {glossary_notes.strip()}
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"请将以下英文地质句子翻译成中文：\n\n{input_text}"}
    ]

    input_ids = deepseek_tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(deepseek_model.device)

    outputs = deepseek_model.generate(
        input_ids,
        max_new_tokens=1024,
        do_sample=False,
        temperature=0.3,
        top_p=0.9,
        repetition_penalty=1.1
    )

    """return deepseek_tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)"""
    full_output = deepseek_tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)

    if "</think>" in full_output:
        translation = full_output.split("</think>")[-1].strip()
    else:
        translation = full_output.strip()

    return translation


def dynamic_threshold(entity_type):
    thresholds = {
        "mineral": 0.95,
        "rock": 0.95,
        "stratum": 0.95,
        "geological time": 0.95,
        "ore deposit": 0.95,
        "location": 0.95,

    }
    return thresholds.get(entity_type, 0.95)


def process_text(input_text):
    annotated_text, entities = analyze_text(input_text)

    similarity_scores = []
    entity_candidates = []
    if entities:
        for entity_text, entity_label in entities:
            threshold = dynamic_threshold(entity_label)
            query_vector = get_vector(entity_text)
            candidates = search_in_milvus(collection, query_vector)

            filtered_candidates = [(word, score) for word, score in candidates if score > threshold]

            if filtered_candidates:
                top_translations = [word for word, _ in filtered_candidates]
                top_scores = [score for _, score in filtered_candidates]
                entity_candidates.append((entity_text, entity_label, top_translations))
                similarity_scores.append((entity_text, top_translations, top_scores))

    translated_text = dynamic_translate(input_text, entity_candidates)

    similarity_data = []
    for entity_text, top_translations, top_scores in similarity_scores:
        similarity_data.extend([
            [entity_text, word, f"{score:.4f}"]
            for word, score in zip(top_translations, top_scores)
        ])

    similarity_df = pd.DataFrame(
        similarity_data,
        columns=["Entity", "Translation", "Similarity Score"]
    ).sort_values("Similarity Score", ascending=False)

    return translated_text, entities, similarity_df


def format_process_details(entities, similarity_df):
    details = []

    if not entities:
        details.append("No geological entities were identified.")
    else:
        details.append("Identified geological entities:")

        for ent_text, ent_label in entities:
            details.append(f"- Entity: {ent_text} | Type: {ent_label}")

    if not similarity_df.empty:
        details.append("\nTerm translation candidates:")
        for _, row in similarity_df.iterrows():

            details.append(f"- {row['Entity']} → {row['Translation']} (Similarity: {row['Similarity Score']})")

    else:
        details.append("\nNo recommended translations found.")

    return "\n".join(details)


def batch_process_file(input_file_path, output_csv_path):
    results = []

    with open(input_file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    for line in tqdm(lines, desc="Processing"):
        try:
            translated_text, entities, similarity_df = process_text(line)
            results.append([line, translated_text])
        except Exception as e:
            print(f"Processing failed: {line}\nError: {e}")
            results.append([line, ""])

    df = pd.DataFrame(results, columns=["original text", "translation result"])
    df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    print(f"The translation results have been saved to: {output_csv_path}")

if __name__ == "__main__":
    input_file_path = "200_en.txt"
    output_csv_path = "tran_rag_en00000.csv"
    batch_process_file(input_file_path, output_csv_path)


