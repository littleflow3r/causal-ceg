#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Version : Python 3.6

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from transformers import BertConfig
from transformers import BertModel


class R_BERT(nn.Module):
    def __init__(self, class_num, user_config):
        super().__init__()
        self.class_num = class_num

        # hyper parameters and others
        bert_config = BertConfig.from_pretrained(user_config.plm_dir)
        self.bert = BertModel.from_pretrained(user_config.plm_dir)
        self.bert_hidden_size = bert_config.hidden_size

        self.max_len = user_config.max_len
        self.dropout_value = user_config.dropout

        # net structures and operations
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(self.dropout_value)

        self.cls_mlp = nn.Linear(
            in_features=self.bert_hidden_size,
            out_features=self.bert_hidden_size,
            bias=True
        )
        self.dense_binary = nn.Linear(
            in_features=self.bert_hidden_size,
            out_features=self.class_num,
            bias=True
        )
        self.criterion = nn.CrossEntropyLoss()
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.cls_mlp.weight)
        init.constant_(self.cls_mlp.bias, 0.)
        # init.xavier_uniform_(self.entity_mlp.weight)
        # init.constant_(self.entity_mlp.bias, 0.)
        init.xavier_uniform_(self.dense_binary.weight)
        init.constant_(self.dense_binary.bias, 0.)

    def bert_layer(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        hidden_output = outputs[0]  # B*L*H
        pooler_output = outputs[1]  # B*H
        return hidden_output, pooler_output

    def forward(self, data, label):
        input_ids = data[:, 0, :].view(-1, self.max_len)
        mask = data[:, 1, :].view(-1, self.max_len)

        attention_mask = mask.gt(0).float()
        token_type_ids = mask.gt(-1).long()
        hidden_output, pooler_output = self.bert_layer(
            input_ids, attention_mask, token_type_ids)

        cls_reps = self.dropout(pooler_output)
        cls_reps = self.tanh(self.cls_mlp(cls_reps))
        #print (cls_reps.shape)

        reps = cls_reps
        reps = self.dropout(reps)
        logits = self.dense_binary(reps)
        loss = self.criterion(logits, label)
        #loss = self.criterion(logits, label.float())
        return loss, logits
