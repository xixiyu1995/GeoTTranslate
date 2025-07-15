import os
import json
import logging
from transformers import BertTokenizer
from utils import cutSentences, commonUtils
import config

logger = logging.getLogger(__name__)


class InputExample:
    def __init__(self, set_type, text, labels=None):
        self.set_type = set_type
        self.text = text
        self.labels = labels


class BaseFeature:
    def __init__(self, token_ids, attention_masks, token_type_ids):
        # BERT 输入
        self.token_ids = token_ids
        self.attention_masks = attention_masks
        self.token_type_ids = token_type_ids


class BertFeature(BaseFeature):
    def __init__(self, token_ids, attention_masks, token_type_ids, labels=None):
        super(BertFeature, self).__init__(
            token_ids=token_ids,
            attention_masks=attention_masks,
            token_type_ids=token_type_ids)
        # labels
        self.labels = labels


class NerProcessor:
    def __init__(self, cut_sent=True, cut_sent_len=256):
        self.cut_sent = cut_sent
        self.cut_sent_len = cut_sent_len

    @staticmethod
    def read_json(file_path):
        """读取 JSON Lines 格式的数据（每行一个 JSON 对象）"""
        raw_examples = []
        with open(file_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        raw_examples.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON解析错误: {line[:50]}... 错误: {e}")
        return raw_examples

    def get_examples(self, raw_examples, set_type):
        examples = []
        empty_entity_count = 0
        for item_idx, item in enumerate(raw_examples):
            # 检查数据项是否完整
            if 'text' not in item:
                logger.warning(f"数据项 [{item_idx}] 缺少 'text' 字段: {item}")
                continue

            text = item['text']
            labels = []  # 默认为空列表

            # 处理实体标签（如果有）
            if 'entities' in item and item['entities']:
                for entity in item['entities']:
                    # 检查所需字段是否存在
                    required_fields = ['entity_label', 'entity_text', 'start_idx']
                    missing_fields = [f for f in required_fields if f not in entity]

                    if missing_fields:
                        logger.warning(f"数据项 [{item_idx}] 实体缺少必要字段 {missing_fields}: {entity}")
                        continue

                    # 转换为需要的格式
                    labels.append([
                        entity['entity_label'],  # 实体类型
                        entity['entity_text'],  # 实体文本
                        entity['start_idx']  # 起始位置
                    ])
            else:
                # 如果没有实体标签，记录日志但不跳过
                if len(text) > 50:
                    logger.debug(f"数据项 [{item_idx}] 无实体标签: {text[:50]}...")
                else:
                    logger.debug(f"数据项 [{item_idx}] 无实体标签: {text}")
                empty_entity_count += 1

            if self.cut_sent:
                # 分句处理
                sentences = cutSentences.cut_sent_for_bert(text, self.cut_sent_len)
                start_index = 0

                for sent in sentences:
                    # 重构分句后的标签
                    sent_labels = []
                    for label in labels:
                        ent_text = label[1]
                        ent_start = label[2]
                        ent_end = ent_start + len(ent_text) - 1

                        # 检查实体是否在当前句子中
                        if start_index <= ent_start <= start_index + len(sent):
                            # 计算实体在新句子中的位置
                            new_start = ent_start - start_index

                            # 确保实体文本在句子中匹配
                            if new_start + len(ent_text) <= len(sent) and sent[new_start:new_start + len(
                                    ent_text)] == ent_text:
                                # 只有实体完全在句子内才添加
                                sent_labels.append([label[0], label[1], new_start])
                            else:
                                # 如果实体文本不匹配，可能是实体被句子拆分，跳过
                                logger.debug(f"实体文本不匹配: {ent_text} in sentence: {sent}")

                    examples.append(InputExample(
                        set_type=set_type,
                        text=sent,
                        labels=sent_labels
                    ))
                    start_index += len(sent)
            else:
                # 不分句处理
                examples.append(InputExample(
                    set_type=set_type,
                    text=text,
                    labels=labels
                ))

        logger.info(f"处理完毕: 共 {len(raw_examples)} 条数据, 其中 {empty_entity_count} 条无实体标签")
        return examples

def convert_bert_example(ex_idx, example: InputExample, tokenizer: BertTokenizer,
                         max_seq_len, ent2id, labels):
    set_type = example.set_type
    raw_text = example.text
    entities = example.labels
    # 文本元组
    callback_info = (raw_text,)
    # 标签字典
    callback_labels = {x: [] for x in labels}
    # _label:实体类别 实体名 实体起始位置
    for _label in entities:
        # print(_label)
        callback_labels[_label[0]].append((_label[1], _label[2]))

    callback_info += (callback_labels,)
    # 序列标注任务 BERT 分词器可能会导致标注偏
    # tokens = commonUtils.fine_grade_tokenize(raw_text, tokenizer)
    tokens = [i for i in raw_text]

    assert len(tokens) == len(raw_text)

    label_ids = None

    # information for dev callback
    # ========================
    label_ids = [0] * len(tokens)

    # tag labels  ent ex. (T1, DRUG_DOSAGE, 447, 450, 小蜜丸)
    for ent in entities:
        # ent: ('PER', '陈元', 0)
        ent_type = ent[0] # 类别

        ent_start = ent[-1] # 起始位置
        ent_end = ent_start + len(ent[1]) - 1

        if ent_start == ent_end:
            label_ids[ent_start] = ent2id['S-' + ent_type]
        else:
            label_ids[ent_start] = ent2id['B-' + ent_type]
            label_ids[ent_end] = ent2id['E-' + ent_type]
            for i in range(ent_start + 1, ent_end):
                label_ids[i] = ent2id['I-' + ent_type]


    if len(label_ids) > max_seq_len - 2:
        label_ids = label_ids[:max_seq_len - 2]

    label_ids = [0] + label_ids + [0]

    # pad
    if len(label_ids) < max_seq_len:
        pad_length = max_seq_len - len(label_ids)
        label_ids = label_ids + [0] * pad_length  # CLS SEP PAD label都为O

    assert len(label_ids) == max_seq_len, f'{len(label_ids)}'

    # ========================
    encode_dict = tokenizer.encode_plus(text=tokens,
                                        max_length=max_seq_len,
                                        padding="max_length",
                                        truncation='longest_first',
                                        return_token_type_ids=True,
                                        return_attention_mask=True)
    tokens = ['[CLS]'] + tokens + ['[SEP]']
    token_ids = encode_dict['input_ids']
    attention_masks = encode_dict['attention_mask']
    token_type_ids = encode_dict['token_type_ids']

    if ex_idx < 3:
        logger.info(f"*** {set_type}_example-{ex_idx} ***")
        print(tokenizer.decode(token_ids[:len(raw_text)+2]))
        logger.info(f'text: {str(" ".join(tokens))}')
        logger.info(f"token_ids: {token_ids}")
        logger.info(f"attention_masks: {attention_masks}")
        logger.info(f"token_type_ids: {token_type_ids}")
        logger.info(f"labels: {label_ids}")
        logger.info('length: ' + str(len(token_ids)))
        # for word, token, attn, label in zip(tokens, token_ids, attention_masks, label_ids):
        #   print(word + ' ' + str(token) + ' ' + str(attn) + ' ' + str(label))
    feature = BertFeature(
        # bert inputs
        token_ids=token_ids,
        attention_masks=attention_masks,
        token_type_ids=token_type_ids,
        labels=label_ids,
    )

    return feature, callback_info


def convert_examples_to_features(examples, max_seq_len, bert_dir, ent2id, labels):
    tokenizer = BertTokenizer(os.path.join(bert_dir, 'vocab.txt'))
    features = []
    callback_info = []

    logger.info(f'Convert {len(examples)} examples to features')

    for i, example in enumerate(examples):
        # 有可能text为空，过滤掉
        if not example.text:
          continue
        feature, tmp_callback = convert_bert_example(
            ex_idx=i,
            example=example,
            max_seq_len=max_seq_len,
            ent2id=ent2id,
            tokenizer=tokenizer,
            labels = labels,
        )
        if feature is None:
            continue
        features.append(feature)
        callback_info.append(tmp_callback)
    logger.info(f'Build {len(features)} features')

    out = (features,)

    if not len(callback_info):
        return out

    out += (callback_info,)
    return out

def get_data(processor, raw_data_path, json_file, mode, ent2id, labels, args):
    raw_examples = processor.read_json(os.path.join(raw_data_path, json_file))
    examples = processor.get_examples(raw_examples, mode)
    data = convert_examples_to_features(examples, args.max_seq_len, args.bert_dir, ent2id, labels)
    save_path = os.path.join(args.data_dir, 'final_data')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    commonUtils.save_pkl(save_path, data, mode)
    return data

def save_file(filename, data ,id2ent):
    features, callback_info = data
    file = open(filename,'w',encoding='utf-8')
    for feature,tmp_callback in zip(features, callback_info):
        text, gt_entities = tmp_callback
        for word, label in zip(text, feature.labels[1:len(text)+1]):
            file.write(word + ' ' + id2ent[label] + '\n')
        file.write('\n')
    file.close()


# 添加新的函数用于处理示例列表而不是文件
def get_data_from_examples(processor, examples, mode, ent2id, labels, args):
    """直接从示例列表生成特征数据"""
    data = convert_examples_to_features(examples, args.max_seq_len, args.bert_dir, ent2id, labels)
    save_path = os.path.join(args.data_dir, 'final_data')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    commonUtils.save_pkl(save_path, data, mode)
    return data

if __name__ == '__main__':
    # 自定义数据集配置
    dataset = "my_data_zh"  # 自定义数据集名称
    args = config.Args().get_parser()
    args.bert_dir = './bert-base-multilingual-cased/'  # 预训练模型路径
    log_dir = os.path.join(args.log_dir)
    os.makedirs(log_dir, exist_ok=True)

    commonUtils.set_logger(os.path.join(log_dir, 'preprocess_zh.log'))


    # 设置数据集路径
    args.data_dir = 'data/my_data_zh'
    args.max_seq_len = 256  # 根据需求调整

    # 创建必要目录
    mid_data_path = os.path.join(args.data_dir, 'mid_data')
    os.makedirs(mid_data_path, exist_ok=True)

    # 1. 处理标签映射
    # 为您的数据集创建标签映射文件
    # 您需要定义实体类型列表，例如: ["ROC", "LOC", ...]
    labels = ["ROC","GST","GTM","MIN","PLA","STR"]  # 根据实际情况添加所有可能的标签

    # 为您的数据集创建实体到ID的映射
    ent2id = {}
    id2ent = {}
    current_id = 0

    # 添加O标签（非实体）
    ent2id['O'] = current_id
    id2ent[current_id] = 'O'
    current_id += 1

    # 添加BIOES风格的标签
    for label in labels:
        for prefix in ['B-', 'I-', 'E-', 'S-']:
            tag = prefix + label
            ent2id[tag] = current_id
            id2ent[current_id] = tag
            current_id += 1

    # 保存标签文件
    labels_path = os.path.join(mid_data_path, 'labels.json')
    with open(labels_path, 'w', encoding='utf-8') as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    # 保存实体到ID的映射
    ent2id_path = os.path.join(mid_data_path, 'nor_ent2id.json')
    with open(ent2id_path, 'w', encoding='utf-8') as f:
        json.dump(ent2id, f, ensure_ascii=False, indent=2)

    # 2. 配置处理器
    processor = NerProcessor(cut_sent=True, cut_sent_len=args.max_seq_len)

    #3.加载全部数据
    all_data_path = "data/NERdata-10803_zh.jsonl"  # 改为您的实际文件路径
    raw_examples = processor.read_json(all_data_path)
    all_examples = processor.get_examples(raw_examples, "all")

    # 4. 随机分割数据集 (80% 训练, 10% 验证, 10% 测试)
    import random

    random.seed(42)  # 设置随机种子以确保可复现性
    random.shuffle(all_examples)

    total = len(all_examples)
    train_count = int(total * 0.8)
    dev_count = int(total * 0.1)
    test_count = total - train_count - dev_count

    # 分割数据集
    train_examples = all_examples[:train_count]
    dev_examples = all_examples[train_count:train_count + dev_count]
    test_examples = all_examples[train_count + dev_count:]

    # 5. 处理分割后的数据集
    # 处理训练集
    train_data = get_data_from_examples(processor, train_examples, "train", ent2id, labels, args)
    save_file(os.path.join(mid_data_path, f"my_data_old_{args.max_seq_len}_cut_train.txt"), train_data, id2ent)

    # 处理开发集
    dev_data = get_data_from_examples(processor, dev_examples, "dev", ent2id, labels, args)
    save_file(os.path.join(mid_data_path, f"my_data_old_{args.max_seq_len}_cut_dev.txt"), dev_data, id2ent)

    # 处理测试集
    test_data = get_data_from_examples(processor, test_examples, "test", ent2id, labels, args)
    save_file(os.path.join(mid_data_path, f"my_data_old_{args.max_seq_len}_cut_test.txt"), test_data, id2ent)

