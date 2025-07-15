from torch import Tensor
import numpy as np
from collections import defaultdict

def get_entities(seq, text, suffix=False):
    """Gets entities from sequence.
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        >>> from seqeval.metrics.sequence_labeling import get_entities
        >>> seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        >>> get_entities(seq)
        [('PER', 0, 1), ('LOC', 3, 3)]
    """
    # for nested list
    if any(isinstance(s, list) for s in seq):
        seq = [item for sublist in seq for item in sublist + ['O']]

    prev_tag = 'O'
    prev_type = ''
    begin_offset = 0
    chunks = []
    for i, chunk in enumerate(seq + ['O']):
        if suffix:
            tag = chunk[-1]
            type_ = chunk.split('-')[0]
        else:
            tag = chunk[0]
            type_ = chunk.split('-')[-1]

        if end_of_chunk(prev_tag, tag, prev_type, type_):
            # chunks.append((prev_type, begin_offset, i-1))
            # 高勇：男，中国国籍，无境外居留权， 高勇：0-2，这里就为text[begin_offset:i]，如果是0-1，则是text[begin_offset:i+1]
            chunks.append((text[begin_offset:i+1],begin_offset,prev_type))
        if start_of_chunk(prev_tag, tag, prev_type, type_):
            begin_offset = i
        prev_tag = tag
        prev_type = type_

    return chunks


def end_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk ended between the previous and current word.
    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.
    Returns:
        chunk_end: boolean.
    """
    chunk_end = False

    if prev_tag == 'E': chunk_end = True
    if prev_tag == 'S': chunk_end = True

    if prev_tag == 'B' and tag == 'B': chunk_end = True
    if prev_tag == 'B' and tag == 'S': chunk_end = True
    if prev_tag == 'B' and tag == 'O': chunk_end = True
    if prev_tag == 'I' and tag == 'B': chunk_end = True
    if prev_tag == 'I' and tag == 'S': chunk_end = True
    if prev_tag == 'I' and tag == 'O': chunk_end = True

    if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
        chunk_end = True

    return chunk_end


def start_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk started between the previous and current word.
    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.
    Returns:
        chunk_start: boolean.
    """
    chunk_start = False

    if tag == 'B': chunk_start = True
    if tag == 'S': chunk_start = True

    if prev_tag == 'E' and tag == 'E': chunk_start = True
    if prev_tag == 'E' and tag == 'I': chunk_start = True
    if prev_tag == 'S' and tag == 'E': chunk_start = True
    if prev_tag == 'S' and tag == 'I': chunk_start = True
    if prev_tag == 'O' and tag == 'E': chunk_start = True
    if prev_tag == 'O' and tag == 'I': chunk_start = True

    if tag != 'O' and tag != '.' and prev_type != type_:
        chunk_start = True

    return chunk_start


# utils/decodeUtils.py

def bioes_decode(sequence, text):
    """
    将BIOS标签序列解码为实体列表

    参数:
    sequence: BIOS标签序列 (如 ["B-PER", "I-PER", "O", "S-LOC"])
    text: 原始文本字符串

    返回:
    entities: 识别的实体字典列表，每个实体包含:
        'type': 实体类型
        'start': 起始位置
        'end': 结束位置
        'text': 实体文本
    """
    entities = []
    entity_start = None
    entity_type = None
    chars = list(text)  # 用于定位实体边界

    for i, tag in enumerate(sequence):
        if tag.startswith("B-"):
            # 遇到新的实体开始
            if entity_start is not None:
                # 保存前一个实体（未结束的）
                entities.append({
                    "type": entity_type,
                    "start": entity_start,
                    "end": i - 1,
                    "text": "".join(chars[entity_start:i])
                })

            # 开始新实体
            entity_start = i
            entity_type = tag.split("-")[1]

        elif tag.startswith("I-"):
            # 实体延续
            if entity_start is None or tag.split("-")[1] != entity_type:
                # 错误情况：I标签没有对应的B标签，或类型不匹配
                entity_start = i
                entity_type = tag.split("-")[1]

        elif tag.startswith("E-") or tag.startswith("S-"):
            # 实体结束或单实体
            if entity_start is not None:
                # 保存完整的实体
                entities.append({
                    "type": entity_type,
                    "start": entity_start,
                    "end": i,
                    "text": "".join(chars[entity_start:i + 1])
                })
                entity_start = None
                entity_type = None
            else:
                # 单实体
                entity_type = tag.split("-")[1]
                entities.append({
                    "type": entity_type,
                    "start": i,
                    "end": i,
                    "text": chars[i]
                })

        elif tag == "O":
            # 非实体结束当前实体
            if entity_start is not None:
                entities.append({
                    "type": entity_type,
                    "start": entity_start,
                    "end": i - 1,
                    "text": "".join(chars[entity_start:i])
                })
                entity_start = None
                entity_type = None

    # 处理未结束的实体（序列末尾）
    if entity_start is not None:
        entities.append({
            "type": entity_type,
            "start": entity_start,
            "end": len(sequence) - 1,
            "text": "".join(chars[entity_start:])
        })

    return entities
