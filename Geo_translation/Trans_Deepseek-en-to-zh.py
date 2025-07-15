import torch
import csv
from transformers import AutoModelForCausalLM, AutoTokenizer

# Device Settings
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the DeepSeek model (for translation)
deepseek_model_path = "DeepSeek-R1-Distill-Qwen-7B"
deepseek_tokenizer = AutoTokenizer.from_pretrained(deepseek_model_path)
deepseek_model = AutoModelForCausalLM.from_pretrained(deepseek_model_path, torch_dtype=torch.float16, device_map="auto")


def translate_text(input_text):
    messages = [{
        "role": "user",
        "content": f"Please translate the following text into Chinese strictly according to the original sentence structure, and ensure that the geological professional terms are accurately translated without adding extra information: {input_text}"
    }]
    input_ids = deepseek_tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(deepseek_model.device)

    outputs = deepseek_model.generate(
        input_ids,
        max_new_tokens=1024,
        do_sample=False
    )

    """response = deepseek_tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
    return response"""

    full_response = deepseek_tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)


    if "</think>" in full_response:
        translated = full_response.split("</think>")[-1].strip()
    else:
        translated = full_response.strip()

    return translated


def process_text(input_text):
    translated_text = translate_text(input_text)
    return input_text, translated_text


def batch_process(input_txt_path, output_csv_path):
    from tqdm import tqdm
    with open(input_txt_path, "r", encoding="utf-8") as infile:
        sentences = [line.strip() for line in infile if line.strip()]

    with open(output_csv_path, "w", newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["original text", "translation result"])


        for idx, sentence in enumerate(tqdm(sentences, desc="Processing", unit="Sentence"), 1):
            try:
                source, translation = process_text(sentence)
                writer.writerow([source, translation])
            except Exception as e:
                writer.writerow([sentence, f"error: {str(e)}"])


if __name__ == "__main__":
    input_file = "100_en.txt"       # Input English text file
    output_file = "Trans_Deepseek-en-to-zh.csv"  # Output translation results to CSV file
    batch_process(input_file, output_file)
    print(f"Processing completed, results saved in: {output_file}")

