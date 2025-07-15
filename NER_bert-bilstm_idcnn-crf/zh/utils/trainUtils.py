# coding=utf-8
import os
import logging
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import torch

logger = logging.getLogger(__name__)


def build_optimizer_and_scheduler(args, model, t_total):
    module = (
        model.module if hasattr(model, "module") else model
    )

    # 差分学习率
    no_decay = ["bias", "LayerNorm.weight"]
    model_param = list(module.named_parameters())

    bert_param_optimizer = []
    crf_param_optimizer = []
    other_param_optimizer = []

    for name, para in model_param:
        space = name.split('.')
        # print(name)
        if space[0] == 'bert_module' or space[0] == "bert":
            bert_param_optimizer.append((name, para))
        elif space[0] == 'crf':
            crf_param_optimizer.append((name, para))
        else:
            other_param_optimizer.append((name, para))

    optimizer_grouped_parameters = [
        # bert other module
        {"params": [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay, 'lr': args.lr},
        {"params": [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': args.lr},

        # crf模块
        {"params": [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay, 'lr': args.crf_lr},
        {"params": [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': args.other_lr},

        # 其他模块，差分学习率
        {"params": [p for n, p in other_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay, 'lr': args.other_lr},
        {"params": [p for n, p in other_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': args.other_lr},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(args.warmup_proportion * t_total), num_training_steps=t_total
    )

    return optimizer, scheduler

"""def save_model(args, model, model_name, global_step):

    output_dir = os.path.join(args.output_dir, '{}'.format(model_name, global_step))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # take care of model distributed / parallel training
    model_to_save = (
        model.module if hasattr(model, "module") else model
    )
    logger.info('Saving model checkpoint to {}'.format(output_dir))
    torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'bilstm_crf_my_data_zh_best.pt'))

def save_model_step(args, model, model_name, global_step):
   
    output_dir = os.path.join(args.output_dir, '{}'.format(model_name, global_step))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # take care of model distributed / parallel training
    model_to_save = (
        model.module if hasattr(model, "module") else model
    )
    logger.info('Saving model & optimizer & scheduler checkpoint to {}.format(output_dir)')
    torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'bilstm_crf_my_data_zh_best.pt'))"""


def save_model(args, model, model_path, global_step):
    """保存最好的验证集效果最好那个模型"""
    try:
        # 确保目录存在
        directory = os.path.dirname(model_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        # 处理并行/分布式训练
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )

        logger.info(f'Saving model checkpoint to {model_path}')

        # 保存模型
        torch.save({
            'global_step': global_step,
            'model_state_dict': model_to_save.state_dict(),
            'args': vars(args)
        }, model_path)
    except Exception as e:
        logger.error(f"Error saving model to {model_path}: {str(e)}")
        raise



def save_model_step(args, model, global_step):
    """根据global_step来保存模型"""
    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # take care of model distributed / parallel training
    model_to_save = (
        model.module if hasattr(model, "module") else model
    )
    logger.info('Saving model & optimizer & scheduler checkpoint to {}.format(output_dir)')
    torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'bilstm_crf_my_data_zh_best.pt'))

def load_model_and_parallel(model, gpu_ids, ckpt_path=None, strict=True):
    """
    加载模型 & 放置到 GPU 中（单卡 / 多卡）
    """
    gpu_ids = gpu_ids.split(',')

    # set to device to the first cuda
    device = torch.device("cpu" if gpu_ids[0] == '-1' else "cuda:" + gpu_ids[0])

    if ckpt_path is not None:
        logger.info(f'Load ckpt from {ckpt_path}')
        checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))

        # 如果是新格式的字典
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict, strict=strict)

    # model.to(device)

    if len(gpu_ids) > 1:
        logger.info('Use multi gpus in: {}'.format(gpu_ids))
        gpu_ids = [int(x) for x in gpu_ids]
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    else:
        logger.info('Use single gpu in: {}'.format(gpu_ids))

    return model, device
