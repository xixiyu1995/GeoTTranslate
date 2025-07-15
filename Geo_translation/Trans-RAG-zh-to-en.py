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
        anns_field="chinese_vector",
        param=search_params,
        limit=top_k,
        output_fields=["english_word"]
    )

    if results and len(results[0]) > 0:
        return [(hit.entity.get("english_word"), hit.distance) for hit in results[0]]

    return []


def analyze_text(input_text):
    """识别地质实体，并返回标注后的文本"""
    instruction = '''你是一个文本实体识别领域的专家，你需要从给定的句子中提取 矿物；岩石；地层；地质年代；地质构造；地名实体. 以 json 格式输出,
        如 {'entity_text': '寒武系', 'entity_label': '地层'} 
        注意: 1. 输出的每一行都必须是正确的 json 字符串. 
        2. 找不到任何实体时, 输出'没有找到任何实体'. '''

    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": f"文本:{input_text}"}
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
            raise ValueError("输出格式错误")

        if not entities:
            return input_text, []

        return input_text, [(ent["entity_text"], ent["entity_label"]) for ent in entities]

    except Exception as e:
        print(f"JSON 解析错误: {e}, response: {response}")
        return input_text, []


def dynamic_translate(input_text, entity_candidates):
    glossary_notes = ""
    if entity_candidates:
        glossary_notes += "\n\n地质术语词汇表（如上下文允许，请优先使用这些英文术语）:\n"
        for entity_text, entity_label, candidates in entity_candidates:
            if candidates:
                glossary_notes += f"- {entity_text} ({entity_label}): {candidates[0]}\n"

    # system prompt
    system_prompt = f"""You are a professional translator specializing in geological literature. Your task is to translate Chinese geological texts into accurate and scientifically consistent English, following these rules:

        1. Preserve the original sentence structure and grammatical logic as much as possible.
        2. Accurately convey domain-specific terms and geological concepts.
        3. Pay close attention to the **semantic types of entities** (e.g., rock, mineral, formation, etc.) listed in the glossary.
        4. Use the suggested translations when appropriate, ensuring the translation fits the entity type and context.
        5. Avoid hallucinations, explanations, or any additional comments—only produce the pure translation.
        6. Maintain a professional, formal style suitable for academic or technical publications.

        Glossary of preferred translations (with entity types):
        {glossary_notes.strip()}
        """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",
         "content": f"Please translate the following Chinese geological sentence into English:\n\n{input_text}"}
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

    full_output = deepseek_tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)

    if "</think>" in full_output:
        translation = full_output.split("</think>")[-1].strip()
    else:
        translation = full_output.strip()

    return translation


def dynamic_threshold(entity_type):
    thresholds = {
        "矿物": 0.95,
        "岩石": 0.95,
        "地层": 0.95,
        "地质年代": 0.95,
        "地质构造": 0.95,
        "地名": 0.95,

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
    input_file_path = "100_zh.txt"  # 替换为包含中文地质文本的文件
    output_csv_path = "Trans-RAG-zh-to-en.csv"
    batch_process_file(input_file_path, output_csv_path)


