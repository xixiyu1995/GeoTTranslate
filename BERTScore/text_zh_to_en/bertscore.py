import os
import json
import torch
import matplotlib.pyplot as plt
import pandas as pd
from bert_score import score


def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def truncate_text(text, max_len=512):
    """Optional truncation to fit model limits."""
    if isinstance(text, list):
        text = " ".join(text)
    return text[:max_len]


def compute_bert_score(json_file, lang='en', use_gpu=True):
    data = load_json(json_file)

    reference_texts = [truncate_text(item["reference_texts"]) for item in data]
    generated_texts = [truncate_text(item["generated_texts"]) for item in data]

    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    P, R, F1 = score(
        generated_texts,
        reference_texts,
        lang=lang,
        model_type="distilbert-base-multilingual-cased",
        num_layers=6,
        verbose=True,
        rescale_with_baseline=True,

        device=device,
        batch_size=64
    )

    return {
        "individual_precision": P.tolist(),
        "individual_recall": R.tolist(),
        "individual_f1": F1.tolist(),
        "average_precision": P.mean().item(),
        "average_recall": R.mean().item(),
        "average_f1": F1.mean().item()
    }


def evaluate_multiple_jsons(json_folder):
    results = {}

    for filename in os.listdir(json_folder):
        if filename.endswith(".json"):
            file_path = os.path.join(json_folder, filename)
            print(f"\nEvaluating {filename} ...")
            result = compute_bert_score(file_path)
            results[filename] = result

    return results


def print_results(results):
    for filename, result in results.items():
        print(f"\n📄 File: {filename}")
        print(f"   🔹 Average Precision: {result['average_precision']:.4f}")
        print(f"   🔹 Average Recall:    {result['average_recall']:.4f}")
        print(f"   🔹 Average F1:        {result['average_f1']:.4f}")
        print("-" * 60)


def plot_bert_scores(results):
    import numpy as np

    file_names = list(results.keys())
    x = np.arange(len(file_names))
    width = 0.25

    avg_p = [results[f]['average_precision'] for f in file_names]
    avg_r = [results[f]['average_recall'] for f in file_names]
    avg_f1 = [results[f]['average_f1'] for f in file_names]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, avg_p, width, label='Precision', color='lightcoral')
    ax.bar(x, avg_r, width, label='Recall', color='lightgreen')
    ax.bar(x + width, avg_f1, width, label='F1', color='skyblue')

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('BERTScore Evaluation for Translation Quality', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(file_names, rotation=30, ha='right')
    ax.set_ylim(0.7, 1.0)
    ax.legend()
    plt.tight_layout()
    plt.show()


def save_results_to_csv(results, output_csv_path="bert_score_summary.csv"):
    rows = []
    for filename, result in results.items():
        rows.append({
            "filename": filename,
            "precision": result["average_precision"],
            "recall": result["average_recall"],
            "f1": result["average_f1"]
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    print(f"✅ Summary saved to {output_csv_path}")


if __name__ == "__main__":
    json_folder = "./"
    results = evaluate_multiple_jsons(json_folder)
    print_results(results)
    plot_bert_scores(results)
    save_results_to_csv(results, "bert_score_output.csv")
