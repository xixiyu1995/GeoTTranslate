import numpy as np
import torch
import argparse
import os,json
import sys
from tqdm import tqdm
import sklearn.preprocessing
from transformers import AutoModel, AutoTokenizer, AutoConfig
from transformers import  get_linear_schedule_with_warmup
from torch.optim import AdamW
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import transformers
import random
from preprocess_en import create_dataloader, get_semeval_data
from bert_utils import compute_loss, get_ent_tags, batch_to_device, compute_f1
from bert_lstm_idcnn_crf import NERNetwork
import logging

# 创建日志目录（如果不存在）
log_dir = 'log'
if not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)

logger=logging.getLogger('main')
logger.setLevel(logging.INFO)
fh=logging.FileHandler('log/log.txt', mode='w')
fh.setLevel(logging.INFO)
ch=logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(lineno)d : %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)


seed=42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# System based
random.seed(seed)
np.random.seed(seed)

device='cuda' if torch.cuda.is_available() else 'cpu'
logger.info("Using device {}".format(device))


def predict(model, test_dataloader, tag_encoder, device):
    logger.info("Evaluating the model...")
    if model.training:
        model.eval()

    predictions = []
    aligned_golden_tags = []

    for batch in test_dataloader:
        batch = batch_to_device(inputs=batch, device=device)
        input_ids, attention_mask, token_type_ids = batch['input_ids'], batch['attention_mask'], batch['token_type_ids']
        target_tags = batch['target_tags']

        with torch.no_grad():
            outputs = model.predict(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        for i, predict_tag_seq in enumerate(outputs):
            preds = tag_encoder.inverse_transform(predict_tag_seq)
            preds = [prediction for prediction, offset in zip(preds.tolist(), batch.get('offsets')[i]) if offset]
            preds = preds[1:-1]  # Remove [CLS] and [SEP]

            golden = target_tags[i].cpu().numpy()
            golden = tag_encoder.inverse_transform(golden)
            golden = [tag for tag, offset in zip(golden.tolist(), batch.get('offsets')[i]) if offset]
            golden = golden[1:-1]  # Remove [CLS] and [SEP]

            predictions.append(preds)
            aligned_golden_tags.append(golden)

    return predictions, aligned_golden_tags


def train(args, train_dataloader, tag_encoder, train_conll_tags, test_conll_tags, test_dataloader):
    n_tags = tag_encoder.classes_.shape[0]
    logger.info("n_tags : {}".format(n_tags))

    print_loss_step = len(train_dataloader) // 5  # 约每80步记录一次损失
    evaluation_steps = 400  # 每400步评估一次
    logger.info(
        "Under an epoch, loss will be output every {} step, and the model will be evaluated every {} step".format(
            print_loss_step, evaluation_steps))

    model = NERNetwork(args, n_tags=n_tags)
    if args.ckpt is not None:
        load_result = model.load_state_dict(torch.load(args.ckpt, map_location='cpu'), strict=False)
        logger.info("Load ckpt to continue training !")
        logger.info("missing and unexpected key : {}".format(str(load_result)))

    model.to(device=device)
    logger.info("Using device : {}".format(device))
    optimizer_parameters = model.parameters()
    optimizer = AdamW(optimizer_parameters, lr=args.learning_rate)
    num_train_steps = int(
        len(train_conll_tags) // args.train_batch_size // args.gradient_accumulation_steps) * args.epochs
    warmup_steps = int(num_train_steps * args.warmup_proportion)
    logger.info("num_train_steps : {}, warmup_proportion : {}, warmup_steps : {}".format(num_train_steps,
                                                                                         args.warmup_proportion,
                                                                                         warmup_steps))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_steps
    )

    global_step = 0
    previous_f1 = -1

    predictions, aligned_test_conll_tags = predict(model=model, test_dataloader=test_dataloader,
                                                   tag_encoder=tag_encoder, device=device)
    f1 = compute_f1(pred_tags=predictions, golden_tags=aligned_test_conll_tags)
    if f1 > previous_f1:
        logger.info("Previous f1 score is {} and current f1 score is {}".format(previous_f1, f1))
        previous_f1 = f1

    total_batches_per_epoch = len(train_dataloader)  # 每轮的批次总数
    total_epochs = args.epochs  # 总轮数

    for epoch in range(total_epochs):
        model.train()
        model.zero_grad()
        training_loss = 0.0

        # 初始化 tqdm 进度条
        with tqdm(total=total_batches_per_epoch,
                  desc=f"Epoch {epoch + 1}/{total_epochs}",
                  unit="batch",
                  mininterval=1.0) as pbar:
            for iteration, batch in enumerate(train_dataloader):
                batch = batch_to_device(inputs=batch, device=device)
                input_ids, attention_mask, token_type_ids = batch['input_ids'], batch['attention_mask'], batch[
                    'token_type_ids']
                loss = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                             target_tags=batch['target_tags'])
                training_loss += loss.item()
                loss.backward()
                if (iteration + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                # 更新进度条
                pbar.update(1)
                # 计算当前 epoch 完成百分比
                epoch_progress = (iteration + 1) / total_batches_per_epoch * 100
                # 计算整体训练进度
                overall_progress = ((epoch * total_batches_per_epoch + (iteration + 1)) / (
                        total_epochs * total_batches_per_epoch)) * 100
                # 更新后缀信息
                pbar.set_postfix({
                    'Epoch Progress': f'{epoch_progress:.1f}%',
                    'Overall': f'{overall_progress:.1f}%',
                    'Loss': f'{training_loss / (iteration + 1):.4f}'
                })

                if (iteration + 1) % print_loss_step == 0:
                    training_loss /= print_loss_step
                    logger.info(
                        "Epoch : {}, global_step : {}/{}, loss_value : {}".format(epoch, global_step, num_train_steps,
                                                                                  training_loss))
                    training_loss = 0.0

                if (iteration + 1) % evaluation_steps == 0:
                    predictions, aligned_test_conll_tags = predict(model=model, test_dataloader=test_dataloader,
                                                                   tag_encoder=tag_encoder, device=device)
                    f1 = compute_f1(pred_tags=predictions, golden_tags=aligned_test_conll_tags)
                    if f1 > previous_f1:
                        torch.save(model.state_dict(), f=os.path.join(args.save_dir, 'pytorch_model.bin'))
                        logger.info(
                            "Previous f1 score is {} and current f1 score is {}, best model has been saved in {}".format(
                                previous_f1, f1, os.path.join(args.save_dir, 'pytorch_model.bin')))
                        previous_f1 = f1
                    else:
                        args.patience -= 1
                        logger.info("Left patience is {}".format(args.patience))
                        if args.patience == 0:
                            logger.info("Total patience is {}, run out of patience, early stop!".format(args.patience))
                            return
                    model.zero_grad()
                    model.train()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_type', choices=['lstm', 'idcnn'], default='lstm', help='选择用于NER的模型结构')

    # Input and output parameters
    parser.add_argument('--model_name_or_path', default='bert-base-multilingual-cased', help='path to the BERT')
    parser.add_argument('--file_path', default='data/OzROCK', help='path to the ner data')
    parser.add_argument('--save_dir', default='saved_models/', help='path to save checkpoints and logs')
    parser.add_argument('--ckpt', default=None, help='Fine tuned model')
    # Training parameters
    parser.add_argument('--learning_rate', default=3e-5, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--train_batch_size', default=32, type=int)
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int)
    parser.add_argument('--lstm_hidden_size', default=256, type=int)
    parser.add_argument('--test_batch_size', default=32, type=int)
    parser.add_argument('--max_grad_norm', default=1, type=int)
    parser.add_argument('--warmup_proportion', default=0.1, type=float)
    parser.add_argument('--max_len', default=256, type=int)
    parser.add_argument('--patience', default=100, type=int)
    # Other parameters
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--take_longest_token', default=False, type=bool)
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        logger.info("save_dir not exists, created!")
        os.makedirs(args.save_dir, exist_ok=True)

    # Use word_idx=0 and entity_idx=1
    train_conll_data = get_semeval_data(split='train', dir=args.file_path, word_idx=0, entity_idx=1)
    test_conll_data = get_semeval_data(split='dev', dir=args.file_path, word_idx=0, entity_idx=1)
    logger.info("train sentences num : {}".format(len(train_conll_data['sentences'])))
    logger.info("test sentences num : {}".format(len(test_conll_data['sentences'])))
    logger.info("Logging some examples...")
    for _ in range(5):
        i = random.randint(0, len(test_conll_data['tags']) - 1)
        sen = test_conll_data['sentences'][i]
        ent = test_conll_data['tags'][i]
        logger.info(f"Sentence {i}:")
        for k in range(len(sen)):
            logger.info(f"Word: {sen[k]}  Tag: {ent[k]}")
        logger.info('-' * 50)

    tag_scheme = get_ent_tags(all_tags=train_conll_data.get('tags'))
    tag_outside = 'O'
    if tag_outside in tag_scheme:
        tag_scheme.remove(tag_outside)
    tag_complete = [tag_outside] + tag_scheme
    logger.info(f"Complete tag set: {tag_complete}, length: {len(tag_complete)}")
    if len(tag_complete) <= 1:
        logger.error("No entity tags found in the dataset. Check the dataset format and entity_idx.")
        raise ValueError("Invalid tag scheme: only 'O' tag found.")
    with open(os.path.join(args.save_dir, 'label.json'), 'w') as f:
        json.dump(obj=' '.join(tag_complete), fp=f)
    logger.info("Tag scheme : {}".format(' '.join(tag_scheme)))
    logger.info("Tag has been saved in {}".format(os.path.join(args.save_dir, 'label.json')))
    tag_encoder = sklearn.preprocessing.LabelEncoder()
    tag_encoder.fit(tag_complete)
    logger.info(f"Tag encoder classes: {tag_encoder.classes_}")

    transformer_tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    transformer_config = AutoConfig.from_pretrained(args.model_name_or_path)

    train_dataloader = create_dataloader(
        sentences=train_conll_data.get('sentences'),
        tags=train_conll_data.get('tags'),
        transformer_tokenizer=transformer_tokenizer,
        transformer_config=transformer_config,
        max_len=args.max_len,
        tag_encoder=tag_encoder,
        tag_outside=tag_outside,
        batch_size=args.train_batch_size,
        num_workers=args.num_workers,
        take_longest_token=args.take_longest_token,
        is_training=True
    )
    test_dataloader = create_dataloader(
        sentences=test_conll_data.get('sentences'),
        tags=test_conll_data.get('tags'),
        transformer_tokenizer=transformer_tokenizer,
        transformer_config=transformer_config,
        max_len=args.max_len,
        tag_encoder=tag_encoder,
        tag_outside=tag_outside,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        take_longest_token=args.take_longest_token,
        is_training=False
    )

    # Args display
    args_config = {}
    for k, v in vars(args).items():
        logger.info(f"{k}:{v}")
        args_config[k] = v

    with open(os.path.join(args.save_dir, 'args_config.dict'), 'w') as f:
        json.dump(args_config, f, ensure_ascii=False)

    train(
        args=args,
        train_dataloader=train_dataloader,
        tag_encoder=tag_encoder,
        train_conll_tags=train_conll_data.get('tags'),
        test_conll_tags=test_conll_data.get('tags'),
        test_dataloader=test_dataloader
    )

if __name__=="__main__":
    main()
