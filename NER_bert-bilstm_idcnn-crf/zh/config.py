import argparse


class Args:
    @staticmethod
    def parse():
        parser = argparse.ArgumentParser()
        return parser

    @staticmethod
    def initialize(parser):

        # args for path
        parser.add_argument('--output_dir', default='./checkpoints',
                            help='the output dir for model checkpoints')

        parser.add_argument('--bert_dir', default='./bert-base-multilingual-cased/',
                            help='bert dir for uer')
        parser.add_argument('--data_dir', default='./data/cner/',
                            help='data dir for uer')
        parser.add_argument('--log_dir', default='./logs/',
                            help='log dir for uer')

        # other args

        parser.add_argument('--num_tags', default=25, type=int,   #标签数量（例如，在NER任务中，不同实体类型的总数）
                            help='number of tags')
        parser.add_argument('--seed', type=int, default=42, help='random seed')

        parser.add_argument('--gpu_ids', type=str, default='0',
                            help='gpu ids to use, -1 for cpu, "0,1" for multi gpu')

        parser.add_argument('--max_seq_len', default=256, type=int)

        parser.add_argument('--eval_batch_size', default=32, type=int)  #eval_batch_size: 评估时的批量大小

        parser.add_argument('--swa_start', default=3, type=int,     #开始使用随机权重平均（Stochastic Weight Averaging）的轮数（epoch）
                            help='the epoch when swa start')

        # train args
        parser.add_argument('--train_epochs', default=5, type=int,
                            help='Max training epoch')

        parser.add_argument('--dropout_prob', default=0.1, type=float,   #模型的dropout概率
                            help='drop out probability')


        parser.add_argument('--eval_steps', default=100, type=int,
                            help='evaluate every x steps')
        # 2e-5
        parser.add_argument('--lr', default=3e-5, type=float,
                            help='bert学习率')
        # 2e-3
        parser.add_argument('--other_lr', default=3e-4, type=float,
                            help='bilstm和多层感知机学习率')
        parser.add_argument('--crf_lr', default=3e-2, type=float,   #3e-2
                            help='条件随机场学习率')
        # 0.5
        parser.add_argument('--max_grad_norm', default=1, type=float,   #梯度裁剪的最大范数（防止梯度爆炸
                            help='max grad clip')

        parser.add_argument('--warmup_proportion', default=0.1, type=float)

        parser.add_argument('--weight_decay', default=0.01, type=float)

        parser.add_argument('--adam_epsilon', default=1e-8, type=float)

        parser.add_argument('--train_batch_size', default=32, type=int)
        parser.add_argument('--use_lstm', type=str, default='True',
                            help='是否使用BiLstm')
        parser.add_argument('--lstm_hidden', default=128, type=int,
                            help='lstm隐藏层大小')
        parser.add_argument('--num_layers', default=1, type=int,
                            help='lstm层数大小')
        parser.add_argument('--dropout', default=0.3, type=float,
                            help='lstm中dropout的设置')
        parser.add_argument('--use_crf', type=str, default='True',
                            help='是否使用Crf')
        parser.add_argument('--use_idcnn', type=str, default='False',
                            help='是否使用Idcnn')
        parser.add_argument('--data_name', type=str, default='my_data_zh ',
                            help='数据集名字')

        parser.add_argument('--model_name', type=str, default='bilstm', help='模型名字')
        #parser.add_argument('--model_name', type=str, default='crf', help='模型名字')
        #parser.add_argument('--model_name', type=str, default='idcnn', help='模型名字')

        parser.add_argument("--use_attention", default="False", type=str, help="是否使用注意力机制")
        parser.add_argument("--attn_heads", default=4, type=int, help="注意力头数")


        parser.add_argument('--use_tensorboard', type=str, default='True',
                            help='是否使用tensorboard可视化')
        parser.add_argument('--use_kd', type=str, default='False',
                            help='是否使用知识蒸馏')
        return parser

    def get_parser(self):
        parser = self.parse()
        parser = self.initialize(parser)
        return parser.parse_args()


"""#当 model_name == "bert"（默认情况）时，使用 bert_ner_model.BertNerModel）
#当 model_name == "bilstm" 时，使用 bert_Bi-LSTM-CRF模型）
#当 model_name == "crf" 时，使用 bert_CRF模型）
#当 model_name == "idcnn" 时，使用 bert_Idcnn-CRF模型）
"""
