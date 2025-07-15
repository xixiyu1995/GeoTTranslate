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
from transformers import TrainerCallback
import os
import swanlab
import numpy as np
from en_utils import dataset_jsonl_transfer, process_func, compute_metrics

# Global Entity Tags
#entity_labels = ['矿物', '岩石', '地层', '地质年代', '地质构造', '地名']
entity_labels = ['mineral', 'rock',  'stratum', 'geological time', 'ore deposit', 'location']

# Set the local model path
model_id = "qwen/Qwen3-1.7B"
model_dir = 'Qwen3-1.7B'

# Load the model directly from local
print("Loading model from local path...")
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map='auto', torch_dtype=torch.bfloat16)
model.enable_input_require_grads()

# Freeze all layers of the pre-trained model
for param in model.parameters():
    param.requires_grad = False

# dataset path
total_dataset_path = 'GNER_data/OzROCK_all.jsonl'
total_jsonl_new_path = 'GNER_data/OzROCK_total.jsonl'

if not os.path.exists(total_jsonl_new_path):
    dataset_jsonl_transfer(total_dataset_path, total_jsonl_new_path)

total_df = pd.read_json(total_jsonl_new_path, lines=True)

total_ds = Dataset.from_pandas(total_df)

# Divide the training set/validation set/test set into a ratio of 8:1:1
split_ratio = total_ds.train_test_split(test_size=0.2, seed=42)
test_val = split_ratio['test'].train_test_split(test_size=0.5, seed=42)

train_dataset = split_ratio['train']
val_dataset = test_val['train']
test_dataset = test_val['test']

print(f"Number of samples in the training set: {len(train_dataset)}")
print(f"Number of samples in the validation set: {len(val_dataset)}")
print(f"Number of samples in the test set: {len(test_dataset)}")

#------------------------------------------------------------
# For quick testing, only take a small amount of data (optional)
"""train_dataset = train_dataset.select(range(100))
val_dataset = val_dataset.select(range(20))
test_dataset = test_dataset.select(range(20))"""
#------------------------------------------------------------
# Data preprocessing
train_dataset = train_dataset.map(process_func, fn_kwargs={'tokenizer': tokenizer},remove_columns=train_dataset.column_names)
val_dataset = val_dataset.map( process_func, fn_kwargs={'tokenizer': tokenizer},remove_columns=val_dataset.column_names)
test_dataset = test_dataset.map(process_func, fn_kwargs={'tokenizer': tokenizer},remove_columns=test_dataset.column_names)

# Make sure that the datasets are not empty
assert len(train_dataset) > 0, "The training set cannot be empty!"
assert len(val_dataset) > 0, "The validation set cannot be empty!"
assert len(test_dataset) > 0, "The test set cannot be empty!"




# Configure Lora
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,  # Dropout rate for Lora layers
    #bias="none"
)

model = get_peft_model(model, config)

# ================= Insert debug statement ================
print("[Training parameter verification]")
#print([name for name, module in model.named_modules() if "lora" in name])
print(f"Trainable parameter quantity: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
# Output model parameter structure to confirm that only a small number of prompt parameters need to be trained
model.print_trainable_parameters()
# ===================================================

# Set training parameters
args = TrainingArguments(
    output_dir='output/GEO-Qwen3-7+8+32+5+1e-4+cosine+0.1',
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    logging_steps=10,
    num_train_epochs=5,
    #save_steps=500,
    save_strategy="epoch",
    #eval_strategy="epoch",
    #eval_strategy="steps",
    #eval_steps=20,
    learning_rate=1e-4,
    #optim="adamw_torch",
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    #lr_scheduler_type="linear",
    #warmup_ratio=0.05,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to='none',
    logging_dir='logs',
    #fp16=True,
    bf16=True,
    #load_best_model_at_end=True,
    #metric_for_best_model='f1',
    #save_total_limit=3,

)

# Create a wrapper function
def wrapped_compute_metrics(eval_pred):
# Pass tokenizer and entity_labels explicitly
    return compute_metrics(eval_pred, tokenizer=tokenizer, entity_labels=entity_labels)

# Custom callback class inherits from TrainerCallback
class CustomPrinterCallback(TrainerCallback):
    def __init__(self, log_dir='./logs'):
        super().__init__()
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.metrics_file = os.path.join(log_dir, "test_en.txt")
        with open(self.metrics_file, 'w') as f:
            f.write("step\ttrain_loss\tlr\teval_loss\tf1\tprecision\trecall\n")

    # Complete callback method implementation (empty method)
    def on_init_end(self, args, state, control, **kwargs):
        pass

    def on_train_begin(self, args, state, control, **kwargs):
        print("Training started!")

    def on_train_end(self, args, state, control, **kwargs):
        print("Training finished!")

    def on_epoch_begin(self, args, state, control, **kwargs):
        pass

    def on_epoch_end(self, args, state, control, **kwargs):
        pass

    def on_step_begin(self, args, state, control, **kwargs):
        pass

    def on_step_end(self, args, state, control, **kwargs):
        pass

    def on_substep_end(self, args, state, control, **kwargs):
        pass

    def on_evaluate(self, args, state, control, **kwargs):
        pass


    def on_predict(self, args, state, control, metrics, **kwargs):
        pass

    def on_save(self, args, state, control, **kwargs):
        pass


    def on_log(self, args, state, control, logs=None, **kwargs):
        if 'loss' in logs and state.global_step % 10 == 0:
            train_loss = logs.get('loss', None)
            lr = logs.get('learning_rate', None)
            print(f"Step {state.global_step}: Train loss={train_loss:.4f}, LR={lr:.6f}")

            # Write to log file
            with open(self.metrics_file, 'a') as f:
                f.write(f"{state.global_step}\t{train_loss:.4f}\t{lr:.6f}\t\t\t\n")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return

        # Use the correct metric key names for the evaluation results
        eval_metrics = {
            'eval_loss': metrics.get('eval_loss', 0.0),
            'f1': metrics.get('eval_f1', 0.0),
            'precision': metrics.get('eval_precision', 0.0),
            'recall': metrics.get('eval_recall', 0.0)
        }

        print("\n" + "=" * 80)
        print(f"Evaluation at step {state.global_step}:")
        print(f"Validation Loss: {eval_metrics['eval_loss']:.4f}")
        if eval_metrics['f1'] is not None:
            print(f"F1 Score: {eval_metrics['f1']:.4f}")
        if eval_metrics['precision'] is not None:
            print(f"Precision: {eval_metrics['precision']:.4f}")
        if eval_metrics['recall'] is not None:
            print(f"Recall: {eval_metrics['recall']:.4f}")
        print("=" * 80 + "\n")

        # Write to log file
        with open(self.metrics_file, 'a') as f:
            if eval_metrics['f1'] is not None and eval_metrics['precision'] is not None and eval_metrics[
                'recall'] is not None:
                f.write(
                    f"{state.global_step}\t\t\t{eval_metrics['eval_loss']:.4f}\t{eval_metrics['f1']:.4f}\t{eval_metrics['precision']:.4f}\t{eval_metrics['recall']:.4f}\n")


# Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    #eval_dataset=test_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    compute_metrics=wrapped_compute_metrics,
    callbacks=[CustomPrinterCallback()]
)


torch.cuda.empty_cache()
trainer.train()

# Evaluate the final model after training
final_results = trainer.evaluate(test_dataset)
# Get the indicator value correctly (add .get() method to avoid KeyError)
print("="*80)
print("Final Evaluation Results:")
print(f"Validation Loss: {final_results.get('eval_loss', 0.0):.4f}")
print(f"F1 Score: {final_results.get('eval_f1', 0.0):.4f}")
print(f"Precision: {final_results.get('eval_precision', 0.0):.4f}")
print(f"Recall: {final_results.get('eval_recall', 0.0):.4f}")
print("="*80)
