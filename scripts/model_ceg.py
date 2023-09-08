#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Version : Python 3.6

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from transformers import BertConfig
from transformers import BertModel


class RE_LCKG(nn.Module):
    def __init__(self, class_num, user_config, lckg_embeddings):
        super().__init__()
        self.class_num = class_num
        
        if lckg_embeddings is not None:
            self.vocab_size, self.embed_dim = lckg_embeddings.shape
            self.embedding = nn.Embedding.from_pretrained(lckg_embeddings, freeze=False)

        # hyper parameters and others
        bert_config = BertConfig.from_pretrained(user_config.plm_dir)
        self.bert = BertModel.from_pretrained(user_config.plm_dir)
        self.bert_hidden_size = bert_config.hidden_size

        self.max_len = user_config.max_len
        self.dropout_value = user_config.dropout

        self.rnn = nn.GRU(self.embed_dim, self.bert_hidden_size, num_layers = 2, bidirectional = True, batch_first = True, dropout = self.dropout_value)

        # net structures and operations
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(self.dropout_value)

        self.cls_mlp = nn.Linear(
            in_features=self.bert_hidden_size,
            out_features=self.bert_hidden_size,
            bias=True
        )

        self.dense_with_lckg = nn.Linear(
            in_features=self.bert_hidden_size*3,
            out_features=self.class_num,
            bias=True
        )

        self.criterion = nn.CrossEntropyLoss()
        #self.criterion = nn.BCEWithLogitsLoss()
        #self.criterion = nn.BCELoss()
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.cls_mlp.weight)
        init.constant_(self.cls_mlp.bias, 0.)
        init.xavier_uniform_(self.dense_with_lckg.weight)
        init.constant_(self.dense_with_lckg.bias, 0.)

        # init.xavier_uniform_(self.entity_mlp.weight)
        # init.constant_(self.entity_mlp.bias, 0.)
        # init.xavier_uniform_(self.dense.weight)
        # init.constant_(self.dense.bias, 0.)
        # init.xavier_uniform_(self.lckg_mlp.weight)
        # init.constant_(self.lckg_mlp.bias, 0.)
        # init.xavier_uniform_(self.lckg_mlp.weight)
        # init.constant_(self.lckg_mlp.bias, 0.)

    def bert_layer(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        hidden_output = outputs[0]  # B*L*H
        pooler_output = outputs[1]  # B*H
        return hidden_output, pooler_output

    def get_entity_reps(self, hidden_output, e_mask):
        e_mask_3 = e_mask.unsqueeze(dim=1).float()  # B*1*L
        sum_reps = torch.bmm(e_mask_3, hidden_output)  # B*1*L * B*L*H -> B*1*H
        sum_reps = sum_reps.squeeze(dim=1)  # B*1*H -> B*H
        entity_lens = e_mask_3.sum(dim=-1).float()  # B*1
        avg_reps = torch.div(sum_reps, entity_lens)
        return avg_reps

    def forward(self, data, label):
        input_ids = data[:, 0, :].view(-1, self.max_len)
        mask = data[:, 1, :].view(-1, self.max_len)
        input_ids_lckg = data[:, 2, :].view(-1, self.max_len)
        #print (input_ids.shape, mask.shape, input_ids_lckg.shape)

        x_embed = self.embedding(input_ids_lckg).float() #[batch size, sent len, emb dim]
        _, hidden = self.rnn(x_embed) #hidden = [n layers * n directions, batch size, emb dim]
        lckg_reps = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)) #from bidirectional both
        #lckg_reps = self.dropout(hidden[-1,:,:]) #hidden = [batch size, hid dim] #non-bidirectional
        #lckg_reps = self.tanh(self.lckg_mlp(lckg_reps))
        #print ('LCKG:', lckg_reps.shape)
        
        attention_mask = mask.gt(0).float()
        token_type_ids = mask.gt(-1).long()
        hidden_output, pooler_output = self.bert_layer(input_ids, attention_mask, token_type_ids)

        #print ('HO, PO:', hidden_output.shape, pooler_output.shape)

        cls_reps = self.dropout(pooler_output)
        cls_reps = self.tanh(self.cls_mlp(cls_reps))
        # print ('CLS:', cls_reps.shape)
        
        reps = torch.cat([cls_reps, lckg_reps], dim=-1)
        # print (reps.shape, cls_reps.shape)
        reps = self.dropout(reps)
        logits = self.dense_with_lckg(reps)
        #print (logits, logits.shape, label)
        loss = self.criterion(logits, label)
        #loss = self.criterion(logits, label.float())
        #print ('loss:', loss)
        return loss, logits
