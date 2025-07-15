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
from utils import dataset_jsonl_transfer, process_func, compute_metrics


entity_labels = ['矿物', '岩石',  '地层', '地质年代', '地质构造', '地名']

#entity_labels = ['ML', 'RK', 'MD', 'SM', 'GA', 'GS', 'FP', 'TM', 'RST', 'GP', 'SF', 'TY', 'GC']



# 设置本地模型路径
model_id = "qwen/Qwen3-1.7B"
model_dir = 'Qwen3-1.7B'

# 直接从本地加载模型
print("Loading model from local path...")
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map='auto', torch_dtype=torch.bfloat16)
model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法

# 冻结预训练模型的所有层
for param in model.parameters():
    param.requires_grad = False

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
train_dataset = split_ratio['train']
val_dataset = test_val['train']
test_dataset = test_val['test']

# 打印各数据集大小
print(f"训练集样本数: {len(train_dataset)}")
print(f"验证集样本数: {len(val_dataset)}")
print(f"测试集样本数: {len(test_dataset)}")

#------------------------------------------------------------
# 为了快速测试，只取少量数据（可选）
"""train_dataset = train_dataset.select(range(100))
val_dataset = val_dataset.select(range(20))
test_dataset = test_dataset.select(range(20))"""
#------------------------------------------------------------

# 数据预处理
train_dataset = train_dataset.map(process_func, fn_kwargs={'tokenizer': tokenizer},remove_columns=train_dataset.column_names)
val_dataset = val_dataset.map( process_func, fn_kwargs={'tokenizer': tokenizer},remove_columns=val_dataset.column_names)
test_dataset = test_dataset.map(process_func, fn_kwargs={'tokenizer': tokenizer},remove_columns=test_dataset.column_names)

# 确保数据集都不为空
assert len(train_dataset) > 0, "训练集不能为空!"
assert len(val_dataset) > 0, "验证集不能为空!"
assert len(test_dataset) > 0, "测试集不能为空!"




# 配置Lora
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    inference_mode=False,  # 训练模式
    r=8,  # Lora 秩
    lora_alpha=32,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1,  # Dropout 比例
    #bias="none"# 偏置项类型，"none"表示没有偏置项，"learnable"表示可学习的偏置项
)

model = get_peft_model(model, config)

# ================= 插入调试语句 =================
print("[训练参数验证]")
#print([name for name, module in model.named_modules() if "lora" in name])
print(f"可训练参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad)}")
# 输出模型参数结构，确认只有少量提示参数需要训练
model.print_trainable_parameters()
# ================================================

# 设置训练参数
args = TrainingArguments(
    output_dir='output/GEO-Qwen3-old-7+8+32+10+1e-4+cosine+0.1',
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    logging_steps=10,
    num_train_epochs=10,
    #save_steps=500,
    save_strategy="epoch",  # 每个epoch保存一次模型
    #eval_strategy="epoch",  # 每个epoch进行评估
    #eval_strategy="steps",  # 按步骤进行评估
    #eval_steps=20,  # 每 步进行一次评估
    learning_rate=1e-4,
    #optim="adamw_torch",  # 显式使用AdamW优化器（PyTorch原生实现）
    lr_scheduler_type="cosine",  # 余弦调度策略
    warmup_ratio=0.1,  # 或者 warmup_steps=500
    #lr_scheduler_type="linear",
    #warmup_ratio=0.05,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to='none',
    logging_dir='./logs',  # 指定日志目录
    #精度混合训练
    #fp16=True,
    bf16=True,
    #load_best_model_at_end=True,  # 训练结束后加载最佳模型
    #metric_for_best_model='f1',  # 使用F1分数作为最佳模型标准
    #save_total_limit=3,  # 最多保存3个模型

)

# 创建包装函数
def wrapped_compute_metrics(eval_pred):
    # 显式传递 tokenizer 和 entity_labels
    return compute_metrics(eval_pred, tokenizer=tokenizer, entity_labels=entity_labels)

# 自定义回调类继承自 TrainerCallback
class CustomPrinterCallback(TrainerCallback):
    def __init__(self, log_dir='./logs'):
        super().__init__()
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.metrics_file = os.path.join(log_dir, "training_metrics_zh_old.txt")
        with open(self.metrics_file, 'w') as f:
            f.write("step\ttrain_loss\tlr\teval_loss\tf1\tprecision\trecall\n")

    # 完整的回调方法实现（空方法）
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
        if 'loss' in logs and state.global_step % 10 == 0:  # 每10步打印一次训练指标
            train_loss = logs.get('loss', None)
            lr = logs.get('learning_rate', None)
            print(f"Step {state.global_step}: Train loss={train_loss:.4f}, LR={lr:.6f}")

            # 写入日志文件
            with open(self.metrics_file, 'a') as f:
                f.write(f"{state.global_step}\t{train_loss:.4f}\t{lr:.6f}\t\t\t\n")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return

        # 使用正确的指标键名
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

        # 写入日志文件
        with open(self.metrics_file, 'a') as f:
            if eval_metrics['f1'] is not None and eval_metrics['precision'] is not None and eval_metrics[
                'recall'] is not None:
                f.write(
                    f"{state.global_step}\t\t\t{eval_metrics['eval_loss']:.4f}\t{eval_metrics['f1']:.4f}\t{eval_metrics['precision']:.4f}\t{eval_metrics['recall']:.4f}\n")


# 创建Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    #eval_dataset=test_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    compute_metrics=wrapped_compute_metrics,
    callbacks=[CustomPrinterCallback()]  # 添加自定义回调
)


torch.cuda.empty_cache()
trainer.train()

# 训练结束后评估最终模型
final_results = trainer.evaluate(test_dataset)
# 正确获取指标值（添加了 .get() 方法避免KeyError）
print("="*80)
print("Final Evaluation Results:")
print(f"Validation Loss: {final_results.get('eval_loss', 0.0):.4f}")
print(f"F1 Score: {final_results.get('eval_f1', 0.0):.4f}")
print(f"Precision: {final_results.get('eval_precision', 0.0):.4f}")
print(f"Recall: {final_results.get('eval_recall', 0.0):.4f}")
print("="*80)