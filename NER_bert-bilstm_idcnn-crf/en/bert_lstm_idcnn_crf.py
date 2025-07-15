import torch
import torch.nn as nn
from transformers import AutoConfig,AutoModel
from bert_utils import match_kwargs
from torchcrf import CRF


class IDCNN(nn.Module):
    def __init__(self, input_size, filters, kernel_size=3, num_block=4):
        super(IDCNN, self).__init__()
        self.layers = [
            {"dilation": 1},
            {"dilation": 1},
            {"dilation": 2}
        ]
        net = nn.Sequential()
        # LayerNorm 的 normalized_shape 为 [filters]
        norms_1 = nn.ModuleList([nn.LayerNorm([filters]) for _ in range(len(self.layers))])

        for i in range(len(self.layers)):
            dilation = self.layers[i]["dilation"]
            single_block = nn.Conv1d(
                in_channels=filters,
                out_channels=filters,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=kernel_size // 2 + dilation - 1
            )
            net.add_module("layer%d" % i, single_block)
            net.add_module("relu", nn.ReLU())
            # 在 LayerNorm 之前转置张量
            net.add_module("layernorm", nn.Sequential(
                nn.LayerNorm([filters])
            ))

        self.linear = nn.Linear(input_size, filters)
        self.idcnn = nn.Sequential()

        norms_2 = nn.ModuleList([nn.LayerNorm([filters]) for _ in range(num_block)])
        for i in range(num_block):
            self.idcnn.add_module("block%i" % i, net)
            self.idcnn.add_module("relu", nn.ReLU())
            # 在 LayerNorm 之前转置张量
            self.idcnn.add_module("layernorm", nn.Sequential(
                nn.LayerNorm([filters])
            ))

    def forward(self, embeddings):
        #print(f"Input embeddings shape: {embeddings.shape}")
        embeddings = self.linear(embeddings)
        #print(f"After linear shape: {embeddings.shape}")
        embeddings = embeddings.permute(0, 2, 1)
        #print(f"After permute shape: {embeddings.shape}")

        output = embeddings
        for module in self.idcnn:
            if isinstance(module, nn.Sequential) and any(isinstance(m, nn.LayerNorm) for m in module):
                #print(f"Before LayerNorm shape: {output.shape}")
                output = output.permute(0, 2, 1)
                #print(f"Before LayerNorm permute shape: {output.shape}")
                output = module(output)
                #print(f"After LayerNorm shape: {output.shape}")
                output = output.permute(0, 2, 1)
                #print(f"After LayerNorm permute shape: {output.shape}")
            else:
                output = module(output)
                #print(f"After module shape: {output.shape}")

        output = output.permute(0, 2, 1)
        #print(f"Output shape: {output.shape}")
        return output

class NERNetwork(nn.Module):
    def __init__(self, args, n_tags):
        super(NERNetwork, self).__init__()
        from transformers import AutoModel

        self.bert = AutoModel.from_pretrained(args.model_name_or_path)
        hidden_size = self.bert.config.hidden_size
        self.model_type = args.model_type

        if args.model_type == 'lstm':
            self.encoder = nn.LSTM(
                input_size=hidden_size,
                hidden_size=args.lstm_hidden_size,
                num_layers=1,
                bidirectional=True,
                batch_first=True
            )
            self.hidden2tag = nn.Linear(args.lstm_hidden_size * 2, n_tags)

        elif args.model_type == 'idcnn':
            self.encoder = IDCNN(
                input_size=hidden_size,  # BERT 的 hidden_size（例如 768）
                filters=args.lstm_hidden_size,  # 128
                kernel_size=3,
                num_block=4
            )
            self.hidden2tag = nn.Linear(args.lstm_hidden_size, n_tags)

        else:
            raise ValueError(f"不支持的模型类型：{args.model_type}")

        self.crf = CRF(num_tags=n_tags, batch_first=True)

    def forward(self, input_ids, attention_mask, token_type_ids, target_tags):
        embeddings = self.bert(input_ids=input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids).last_hidden_state
        if self.model_type == 'lstm':
            encoder_outputs, _ = self.encoder(embeddings)
        else:
            encoder_outputs = self.encoder(embeddings)

        emissions = self.hidden2tag(encoder_outputs)
        loss = -self.crf(emissions, target_tags, mask=attention_mask.bool(), reduction='mean')
        return loss

    def predict(self, input_ids, attention_mask, token_type_ids):
        embeddings = self.bert(input_ids=input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids).last_hidden_state
        if self.model_type == 'lstm':
            encoder_outputs, _ = self.encoder(embeddings)
        else:
            encoder_outputs = self.encoder(embeddings)

        emissions = self.hidden2tag(encoder_outputs)
        predictions = self.crf.decode(emissions, mask=attention_mask.bool())
        return predictions


