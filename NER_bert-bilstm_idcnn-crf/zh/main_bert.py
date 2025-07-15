
import os
import logging
import numpy as np
import torch

# 添加以下行：完全禁用 TorchDynamo 和编译优化
torch._dynamo.config.disable = True
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.verbose = False

# 设置此环境变量以跳过 Triton
os.environ["DISABLE_TRITON"] = "1"


import os
import logging
import numpy as np
import torch
from utils import commonUtils, metricsUtils, decodeUtils, trainUtils
import config
import dataset
from preprocess import BertFeature
import bert_ner_model
from torch.utils.data import DataLoader, RandomSampler
from transformers import BertTokenizer
from tensorboardX import SummaryWriter
from collections import defaultdict
from utils.decodeUtils import bioes_decode

"""if torch.__version__.startswith("2."):
    import torch._dynamo

    torch._dynamo.config.suppress_errors = True"""

args = config.Args().get_parser()
commonUtils.set_seed(args.seed)
logger = logging.getLogger(__name__)

special_model_list = ['bilstm', 'crf', 'idcnn']

if args.use_tensorboard == "True":
    writer = SummaryWriter(log_dir='./tensorboard')


class BertForNer:
    def __init__(self, args, train_loader, dev_loader, test_loader, idx2tag):
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.args = args
        self.idx2tag = idx2tag
        if args.model_name.split('_')[0] not in special_model_list:
            model = bert_ner_model.BertNerModel(args)
        else:
            model = bert_ner_model.NormalNerModel(args)
        self.model, self.device = trainUtils.load_model_and_parallel(model, args.gpu_ids)
        self.model.to(self.device)
        """if torch.__version__.startswith("2."):
            self.model = torch.compile(self.model)"""
        self.t_total = len(self.train_loader) * args.train_epochs
        self.optimizer, self.scheduler = trainUtils.build_optimizer_and_scheduler(args, model, self.t_total)

    def train(self):
        # Train
        global_step = 0
        self.model.zero_grad()
        eval_steps = self.args.eval_steps  # 每多少个step进行验证
        best_f1 = 0.0
        for epoch in range(self.args.train_epochs):
            for step, batch_data in enumerate(self.train_loader):
                self.model.train()
                for key in batch_data.keys():
                    if key != 'texts':
                        batch_data[key] = batch_data[key].to(self.device)

                outputs = self.model(
                    batch_data['token_ids'],
                    batch_data['attention_masks'],
                    batch_data['token_type_ids'],
                    batch_data['labels']
                )
                loss = outputs[0]

                if len(outputs) > 1:
                    logits = outputs[1]

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.model.zero_grad()

                logger.info(f'【train】 epoch:{epoch} {global_step}/{self.t_total} loss:{loss.item():.4f}')

                if self.args.use_tensorboard == "True":
                    writer.add_scalar('train/loss', loss.item(), global_step)

                global_step += 1

                if global_step % eval_steps == 0:
                    dev_loss, precision, recall, f1_score = self.dev()
                    if self.args.use_tensorboard == "True":
                        writer.add_scalar('dev/loss', dev_loss, global_step)
                        writer.add_scalar('dev/f1', f1_score, global_step)

                    logger.info(
                        f'[eval] loss:{dev_loss:.4f} precision={precision:.4f} recall={recall:.4f} f1_score={f1_score:.4f}')

                    if f1_score > best_f1:
                        model_path = os.path.join(self.args.output_dir,
                                                  f'{self.args.model_name}_{self.args.data_name}_best.pt')
                        trainUtils.save_model(self.args, self.model, model_path, global_step)
                        best_f1 = f1_score

    def dev(self):
        self.model.eval()
        tot_dev_loss = 0.0
        all_predictions = []
        all_labels = []
        with torch.no_grad():
            for dev_batch_data in self.dev_loader:
                for key in dev_batch_data.keys():
                    if key != 'texts':
                        dev_batch_data[key] = dev_batch_data[key].to(self.device)

                outputs = self.model(
                    dev_batch_data['token_ids'],
                    dev_batch_data['attention_masks'],
                    dev_batch_data['token_type_ids'],
                    dev_batch_data['labels']
                )

                dev_loss = outputs[0]
                tot_dev_loss += dev_loss.item()

                # 收集预测结果
                if self.args.use_crf == 'True':
                    predictions = outputs[1]
                else:
                    logits = outputs[1].detach().cpu().numpy()
                    predictions = np.argmax(logits, axis=2).tolist()

                all_predictions.extend(predictions)
                all_labels.extend(dev_batch_data['labels'].cpu().numpy())

        # 计算验证集性能
        precision, recall, f1 = self.compute_metrics(all_predictions, all_labels)
        return tot_dev_loss / len(self.dev_loader), precision, recall, f1

    def compute_metrics(self, predictions, labels):
        """计算基于实体级别的指标"""

        # 步骤1：将模型预测的标签序列解码为实体列表
        all_entities = []
        all_predictions = []

        # 步骤2：遍历每个样本
        for pred_seq, true_seq in zip(predictions, labels):
            # 确保pred_seq和true_seq是NumPy数组
            pred_seq = np.array(pred_seq)
            true_seq = np.array(true_seq)

            # 移除padding部分（-100）
            mask = true_seq != -100
            real_pred_seq = pred_seq[mask]
            real_true_seq = true_seq[mask]

            # 将标签ID转换为标签名称
            pred_tags = [self.idx2tag[idx] for idx in real_pred_seq]
            true_tags = [self.idx2tag[idx] for idx in real_true_seq]

            # 创建临时占位符文本（与序列长度匹配）
            dummy_text = ''.join(['a' for _ in range(len(real_true_seq))])

            # 使用 bioes_decode 进行解码
            pred_entities = bioes_decode(pred_tags, dummy_text)
            true_entities = bioes_decode(true_tags, dummy_text)

            # 格式化实体为元组列表 (起始位置, 结束位置, 实体类型)
            pred_list = [(ent['start'], ent['end'], ent['type']) for ent in pred_entities]
            true_list = [(ent['start'], ent['end'], ent['type']) for ent in true_entities]

            # 添加到总体列表
            all_entities.append(true_list)
            all_predictions.append(pred_list)

        # 步骤3：初始化各实体类型的TP, FP, FN
        type_metrics = defaultdict(lambda: np.array([0, 0, 0]))  # 格式: {type: [tp, fp, fn]}

        # 步骤4：计算每个实体的指标
        for true_list, pred_list in zip(all_entities, all_predictions):
            # 按实体类型分组
            true_types = defaultdict(list)
            pred_types = defaultdict(list)

            for start, end, ent_type in true_list:
                true_types[ent_type].append((start, end))
            for start, end, ent_type in pred_list:
                pred_types[ent_type].append((start, end))

            # 为每个实体类型计算指标
            for ent_type in set(true_types.keys()) | set(pred_types.keys()):
                gt_list = true_types.get(ent_type, [])
                pred_list = pred_types.get(ent_type, [])

                # 使用metricsUtils计算当前类型的指标
                metrics = metricsUtils.calculate_metric(gt_list, pred_list)
                type_metrics[ent_type] += metrics

        # 步骤5：计算宏平均
        macro_precision, macro_recall, macro_f1 = 0.0, 0.0, 0.0
        valid_types = 0

        for ent_type, metrics in type_metrics.items():
            # 跳过非实体类型
            if ent_type == "O" or metrics.sum() == 0:
                continue

            # 计算当前类型的指标
            precision, recall, f1 = metricsUtils.get_p_r_f(*metrics)

            # 累加以计算宏平均
            macro_precision += precision
            macro_recall += recall
            macro_f1 += f1
            valid_types += 1

        # 步骤6：计算宏平均
        if valid_types > 0:
            macro_precision /= valid_types
            macro_recall /= valid_types
            macro_f1 = 2 * macro_precision * macro_recall / (macro_precision + macro_recall) if (
                                                                                                        macro_precision + macro_recall) > 0 else 0
        else:
            macro_precision, macro_recall, macro_f1 = 0.0, 0.0, 0.0

        return macro_precision, macro_recall, macro_f1

    def predict(self, raw_text, model_path=None):
        """预测单个文本"""
        if model_path is None:
            model = self.model
            device = self.device
        else:
            if self.args.model_name.split('_')[0] not in special_model_list:
                model = bert_ner_model.BertNerModel(self.args)
            else:
                model = bert_ner_model.NormalNerModel(self.args)
            model, device = trainUtils.load_model_and_parallel(model, self.args.gpu_ids, model_path)
            model.to(device)

        model.eval()
        with torch.no_grad():
            tokenizer = BertTokenizer(
                os.path.join(self.args.bert_dir, 'vocab.txt'))
            # tokens = commonUtils.fine_grade_tokenize(raw_text, tokenizer)
            tokens = [i for i in raw_text]
            encode_dict = tokenizer.encode_plus(
                text=tokens,
                max_length=self.args.max_seq_len,
                padding='max_length',
                truncation='longest_first',
                is_pretokenized=True,
                return_token_type_ids=True,
                return_attention_mask=True
            )

            token_ids = torch.tensor([encode_dict['input_ids']]).to(device)
            attention_masks = torch.tensor([encode_dict['attention_mask']]).to(device)
            token_type_ids = torch.tensor([encode_dict['token_type_ids']]).to(device)

            logits = model(token_ids, attention_masks, token_type_ids, None)

            if self.args.use_crf == 'True':
                predictions = logits[0]
            else:
                predictions = logits.detach().cpu().numpy()
                predictions = np.argmax(predictions, axis=2)[0]

            # 将索引转换为标签
            token_predictions = [self.idx2tag[idx] for idx in predictions[1:1 + len(tokens)]]

            # 获取原始文本中的实体
            text = "".join(tokens)
            entities = bioes_decode(token_predictions, text)

            logger.info(f"Text: {text}")
            logger.info(f"Predictions: {list(zip(tokens, token_predictions))}")
            logger.info(f"Entities: {entities}")

            return entities


if __name__ == '__main__':
    # 设置数据集名称（应与预处理设置一致）
    data_name = "my_data_en"  # 您的自定义数据集名称

    # 设置模型名称和结构
    if args.model_name == 'bilstm':
        model_name = "bilstm_crf"
        args.use_lstm = "True"
        args.use_idcnn = "False"
        args.use_crf = "True"
    elif args.model_name == 'crf':
        model_name = "crf"
        args.use_lstm = "False"
        args.use_idcnn = "False"
        args.use_crf = "True"
    elif args.model_name == "idcnn":
        model_name = "idcnn_crf"
        args.use_idcnn = "True"
        args.use_lstm = "False"
        args.use_crf = "True"
    else:
        # 默认为基本BERT模型
        model_name = "bert"
        args.use_lstm = "False"
        args.use_idcnn = "False"
        args.use_crf = "False"

    # 设置数据集目录
    args.data_name = data_name
    args.data_dir = f'./data/{data_name}'

    # 设置模型名称
    args.model_name = model_name

    # 配置日志
    log_file = os.path.join(args.log_dir, f'{model_name}_{data_name}.log')
    commonUtils.set_logger(log_file)
    logger.info(f"Starting training for {model_name} on {data_name} dataset")

    # 加载标签映射
    other_path = os.path.join(args.data_dir, 'mid_data')

    # 正确调用方式
    ent2id = commonUtils.read_json(other_path, 'nor_ent2id')
    labels = commonUtils.read_json(other_path, 'labels')

    # 创建ID到标签的映射
    idx2tag = {idx: tag for tag, idx in ent2id.items()}

    # 设置标签数量
    args.num_tags = len(ent2id)
    logger.info(f"Number of tags: {args.num_tags}")

    # 加载预处理数据
    data_path = os.path.join(args.data_dir, 'final_data')

    # 假设read_pkl也是类似设计
    train_features = commonUtils.read_pkl(data_path, 'train')[0] if hasattr(commonUtils, 'read_pkl') else []
    dev_features = commonUtils.read_pkl(data_path, 'dev')[0] if hasattr(commonUtils, 'read_pkl') else []
    test_features = commonUtils.read_pkl(data_path, 'test')[0] if hasattr(commonUtils, 'read_pkl') else []

    # 如果没有read_pkl函数，用其他方式加载
    if not train_features:
        # 替代加载方式
        import pickle

        with open(os.path.join(data_path, 'train.pkl'), 'rb') as f:
            train_features = pickle.load(f)[0]
        with open(os.path.join(data_path, 'dev.pkl'), 'rb') as f:
            dev_features = pickle.load(f)[0]
        with open(os.path.join(data_path, 'test.pkl'), 'rb') as f:
            test_features = pickle.load(f)[0]

    # 创建数据集
    train_dataset = dataset.NerDataset(train_features)
    dev_dataset = dataset.NerDataset(dev_features)
    test_dataset = dataset.NerDataset(test_features)

    # 创建数据加载器
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.train_batch_size,
        sampler=RandomSampler(train_dataset),
        num_workers=2
    )
    dev_loader = DataLoader(
        dataset=dev_dataset,
        batch_size=args.eval_batch_size,
        num_workers=2
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.eval_batch_size,
        num_workers=2
    )

    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)

    # 保存配置参数
    commonUtils.save_json(args.output_dir, vars(args), 'config')

    # 初始化训练器
    bert_ner = BertForNer(args, train_loader, dev_loader, test_loader, idx2tag)

    # 训练模型
    bert_ner.train()

    # 加载最佳模型并测试
    model_path = os.path.join(args.output_dir, f'{model_name}_{data_name}_best.pt')
    logger.info(f"Evaluating best model from {model_path}")

    # 测试集评估
    bert_ner.model, _ = trainUtils.load_model_and_parallel(
        bert_ner.model, args.gpu_ids, model_path
    )
    dev_loss, precision, recall, f1 = bert_ner.dev()
    logger.info(f"[Test] loss:{dev_loss:.4f} precision={precision:.4f} recall={recall:.4f} f1_score={f1:.4f}")

    # 预测示例
    if args.example_text:
        logger.info(f"Predicting example text: {args.example_text}")
        entities = bert_ner.predict(args.example_text)
        logger.info(f"Predicted entities: {entities}")
    else:
        default_text = "角闪石，辉石不均分布，石榴子石为铁铝质的钙铝榴石，其晶体成分及化学式见表。"  # 替换为您的领域相关文本
        entities = bert_ner.predict(default_text)
        logger.info(f"Predicted entities: {entities}")