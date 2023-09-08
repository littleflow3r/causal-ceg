#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Version : Python 3.6


import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers.tokenization_bert import BertTokenizer
from tqdm import tqdm
import random

class RelationLoader(object):
    def __init__(self, config):
        self.data_dir = config.data_dir

    def __load_relation(self):
        relation_file = os.path.join(self.data_dir, 'relation2id.txt')
        rel2id = {}
        id2rel = {}
        with open(relation_file, 'r', encoding='utf-8') as fr:
            for line in fr:
                relation, id_s = line.strip().split()
                id_d = int(id_s)
                rel2id[relation] = id_d
                id2rel[id_d] = relation
        return rel2id, id2rel, len(rel2id)

    def get_relation(self):
        return self.__load_relation()

class Tokenizer(object):
    def __init__(self, config):
        self.data_dir = config.data_dir
        self.plm_dir = config.plm_dir
        self.tokenizer, self.special_tokens = self.load_tokenizer()

    def load_tokenizer(self):
        tokenizer = BertTokenizer.from_pretrained(self.plm_dir)
        # entity marker: <e1>, </e1> -> `$`, <e2>, </e2> -> `#`
        special_tokens = ['$', '#']
        special_tokens_dict = {'additional_special_tokens': special_tokens}
        tokenizer.add_special_tokens(special_tokens_dict)
        return tokenizer, special_tokens

    def build_vocab(self):
        vocab = set()
        filelist = ['train2', 'test2', 'dev2']
        for filename in filelist:
            src_file = os.path.join(self.data_dir, '{}.json'.format(filename))
            if not os.path.isfile(src_file):
                continue
            print('BERT DATA: get the result of tokenization from %s' % src_file)
            with open(src_file, 'r', encoding='utf-8') as fr:
                for line in fr:
                    sentence = json.loads(line.strip())['sentence']
                    for token in sentence:
                        if token in ['<e1>', '</e1>', '<e2>', '</e2>']:
                            continue
                        vocab.add(token)
        return vocab
    
    def get_vocab(self):
        vocab_set = self.build_vocab()
        #print ('vocab_set:', vocab_set, len(vocab_set)) #'recipients.', 'nocturnal', 'SIDS', 'FFA', 'DR/DQ', 'M416V', 'reproduction,',  9415
        vocab_dict = {}
        extra_tokens = ['[CLS]', '[SEP]', '[PAD]']
        for token in extra_tokens + self.special_tokens:
            vocab_dict[token] = [self.tokenizer.convert_tokens_to_ids(token)]
            #print (token, vocab_dict[token]) #[CLS] [101] [SEP] [102] [PAD] [0] $ [109] # [108]

        for token in vocab_set:
            token = token.lower()
            if token in vocab_dict.keys():
                continue
            token_res = self.tokenizer.tokenize(token)
            if len(token_res) < 1:
                token_idx_list = [self.tokenizer.convert_tokens_to_ids('[UNK]')]
            else:
                token_idx_list = self.tokenizer.convert_tokens_to_ids(token_res)
            vocab_dict[token] = token_idx_list
        #print ('vocab_dict:', vocab_dict) # 'showing': [4000], 'down-regulation': [1205, 118, 8585], 'a114v': [170, 49967
        return vocab_dict

class Tokenizer_LCKG(object):
    def __init__(self, config):
        self.data_dir = config.data_dir
        self.plm_dir = config.plm_dir
        #self.lckg_dir = 'resource/lckg/aimedlcgent_ckv_q3.txt'
        self.bert_tokenizer = BertTokenizer.from_pretrained(self.plm_dir)
        self.max_len = config.max_len
        self.vocab_set = self.build_vocab_lckg()

    def build_vocab_lckg(self):
        vocab = set()
        filelist = ['train2', 'dev2', 'test2']
        for filename in filelist:
            src_file = os.path.join(self.data_dir, '{}.json'.format(filename))
            if not os.path.isfile(src_file):
                continue
            print('LCKG: get the result of tokenization from %s' % src_file)
            with open(src_file, 'r', encoding='utf-8') as fr:
                for line in fr:
                    sentence = json.loads(line.strip())['sentence']
                    sentence_bert = self.bert_tokenizer.tokenize(' '.join(sentence))
                    for token in sentence_bert:
                        if token in ['<e1>', '</e1>', '<e2>', '</e2>']:
                            continue
                        vocab.add(token)
        return vocab

    def get_vocab_lckg(self):
        max_len = self.max_len
        self.word2idx = {}

        # Add <pad> and <unk> tokens to the vocabulary
        self.word2idx['[PAD]'] = 0
        self.word2idx['[CLS]'] = 1
        self.word2idx['[SEP]'] = 2
        self.word2idx['[UNK]'] = 3
        self.word2idx['$'] = 4
        self.word2idx['#'] = 5
        
        #vocab_set = self.build_vocab_lckg()
        #print (vocab_set)
        # Building our vocab from the corpus starting from index 6
        idx = 5
        for token in self.vocab_set:
            if token not in self.word2idx:
                idx += 1
                self.word2idx[token] = idx
            
        # print ('vset:', len(self.vocab_set), len(self.vocab_set)+6, len(self.word2idx))
        # print ('word2idx k,v:', max(self.word2idx, key=self.word2idx.get), max(self.word2idx.values()) )

        vocab_dict = {}
        extra_tokens = ['[CLS]', '[SEP]', '[PAD]', '[UNK]']
        special_tokens = ['$', '#']
        for token in extra_tokens + special_tokens:
            vocab_dict[token] = [self.word2idx.get(token)]

        for token in self.vocab_set:
            token = token.lower()
            if token in vocab_dict.keys():
                continue
            vocab_dict[token] = [self.word2idx.get(token)]
        return vocab_dict, self.word2idx

class LCKG_Embeddings(object):
    def __init__(self, lckg_dir, word2idxx):
        #self.lckg_dir = 'resource/lckg/aimedlcgent_ckv_q3.txt'
        self.lckg_dir = lckg_dir
        self.word2idxx = word2idxx

    def load_pretrained_lckg(self):
        fname = self.lckg_dir
        print("LCKG: Loading pretrained LCKG vectors from ", fname)
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        #n, d = map(int, fin.readline().split())
        d = 400

        # Initilize random embeddings
        lckg_embeddings = np.random.uniform(-0.25, 0.25, (len(self.word2idxx), d))
        # print ('LCKG_Embeddings:', lckg_embeddings.shape)
        extra_tokens = ['[CLS]', '[SEP]', '[PAD]', '[UNK]']
        special_tokens = ['$', '#']
        for token in extra_tokens + special_tokens:
            lckg_embeddings[self.word2idxx[token]] = np.zeros((d,))

        # Load pretrained vectors
        count = 0
        for line in tqdm(fin):
            tokens = line.rstrip().split(' ')
            if len(tokens) == 401:
                word = tokens[0]
                if word in self.word2idxx:
                    # if self.word2idxx[word] == 16063:
                    #     print ('LCKG_Embeddings:', count, word, self.word2idxx[word])
                    lckg_embeddings[self.word2idxx[word]] = np.array(tokens[1:], dtype=np.float32)
                    count += 1

        print(f"LCKG: There are {count} / {len(self.word2idxx)} pretrained vectors found.")
        return lckg_embeddings

class MyCorpus(object):
    def __init__(self, rel2id, config):
        self.lckg_dir = config.lckg_dir
        self.rel2id = rel2id
        self.class_num = len(rel2id)
        self.max_len = config.max_len
        self.data_dir = config.data_dir
        self.cache_dir = config.cache_dir
        self.tokenizer = Tokenizer(config)
        self.vocab = None

        self.tokenizer_lckg = Tokenizer_LCKG(config)
        # self.vocab_lckg = None
        self.vocab_lckg, self.word2idxlckg = self.tokenizer_lckg.get_vocab_lckg()
        self.plm_dir = config.plm_dir
        self.bert_tokenizer = BertTokenizer.from_pretrained(self.plm_dir)

    def __symbolize_sentence(self, sentence):
        """
            Args:
                sentence (list)
            Return:
                sent(ids): [CLS] ... $ e1 $ ... # e2 # ... [SEP] [PAD]
                mask     :   1    3  4  4 4  3  5  5 5  3    2     0
        """
        try:
            assert '<e1>' in sentence
            assert '<e2>' in sentence
            assert '</e1>' in sentence
            assert '</e2>' in sentence
        except:
            print (sentence)
        assert '<e1>' in sentence
        assert '<e2>' in sentence
        assert '</e1>' in sentence
        assert '</e2>' in sentence
        sentence_token = []
        sentence_mask = []
        sentence_token_lckg = []
        # postion of e1 (p11, p12), e2 (p21, p22) after tokenization
        p11 = p12 = p21 = p22 = -1
        for token in sentence:
            token = token.lower()
            if token == '<e1>':
                p11 = len(sentence_token)
                sentence_token += self.vocab['$']
                sentence_token_lckg += self.vocab_lckg['$']
            elif token == '</e1>':
                p12 = len(sentence_token)
                sentence_token += self.vocab['$']
                sentence_token_lckg += self.vocab_lckg['$']
            elif token == '<e2>':
                p21 = len(sentence_token)
                sentence_token += self.vocab['#']
                sentence_token_lckg += self.vocab_lckg['#']
            elif token == '</e2>':
                p22 = len(sentence_token)
                sentence_token += self.vocab['#']
                sentence_token_lckg += self.vocab_lckg['#']
            else:
                bert_token = self.vocab[token]
                lckg_token = self.vocab_lckg.get(token)
                if not lckg_token:
                    lckg_token = []
                    tt = self.bert_tokenizer.tokenize(token)
                    for t in tt:
                        ttt = self.vocab_lckg.get(t)
                        if not ttt:
                            ttt = self.vocab_lckg.get('[UNK]')
                        lckg_token.append(ttt[0])
                assert len(bert_token) == len(lckg_token)
                sentence_token += bert_token
                sentence_token_lckg += lckg_token 
                #print ('token, btoken, lckg_token:', token, bert_token, lckg_token) # c1772t [172, 16770, 32910, 1204] [None, None, None, None]

        assert len(sentence_token) == len(sentence_token_lckg)
        # for x,y in zip(sentence_token, sentence_token_lckg):
        #     print (x, y)
        # sys.exit()
        sentence_mask = [3] * len(sentence_token)
        sentence_mask[p11: p12+1] = [4] * (p12 - p11 + 1)
        sentence_mask[p21: p22+1] = [5] * (p22 - p21 + 1)

        if len(sentence_token) > self.max_len-2:
            sentence_token = sentence_token[:self.max_len-2]
            sentence_mask = sentence_mask[:self.max_len-2]
            sentence_token_lckg = sentence_token_lckg[:self.max_len-2]

        pad_length = self.max_len - 2 - len(sentence_token)
        mask = [1] + sentence_mask + [2] + [0] * pad_length
        input_ids = self.vocab['[CLS]'] + sentence_token + self.vocab['[SEP]']
        input_ids += self.vocab['[PAD]'] * pad_length

        input_ids_lckg = self.vocab_lckg['[CLS]'] + sentence_token_lckg + self.vocab_lckg['[SEP]']
        input_ids_lckg += self.vocab_lckg['[PAD]'] * pad_length

        assert len(mask) == self.max_len
        assert len(input_ids) == self.max_len
        assert len(input_ids_lckg) == self.max_len

        unit_lckg = np.asarray([input_ids, mask, input_ids_lckg], dtype=np.int64)
        unit = np.asarray([input_ids, mask], dtype=np.int64)
        #print (unit.shape, unit_lckg.shape)
        unit = np.reshape(unit, newshape=(1, 2, self.max_len))
        unit_lckg = np.reshape(unit_lckg, newshape=(1, 3, self.max_len))
        #print (unit.shape, unit_lckg.shape)
        return unit_lckg

    def __load_data(self, filetype):
        # data_cache = os.path.join(self.cache_dir, '{}.pkl'.format(filetype))
        # if os.path.exists(data_cache):
        #     data, labels = torch.load(data_cache)
        # else:
        if self.vocab is None:
            self.vocab = self.tokenizer.get_vocab()
            # print ('vocablen:', len(self.vocab)) #9142
        src_file = os.path.join(self.data_dir, '{}.json'.format(filetype))
        data = []
        labels = []
        with open(src_file, 'r', encoding='utf-8') as fr:
            for line in fr:
                line = json.loads(line.strip())
                label = line['relation']
                sentence = line['sentence']
                label_idx = self.rel2id[label]
                #print ('A:',sentence, len(sentence), '#',label, label_idx) #original sentence
                one_sentence = self.__symbolize_sentence(sentence)
                #print ('Ax', one_sentence) #sentence idx?
                data.append(one_sentence)
                labels.append(label_idx)
                #print ('datalabels', data, labels)
                #sys.exit()
        data_labels = [data, labels]
        #torch.save(data_labels, data_cache)
        print ('DATA:', filetype, len(data))
        return data, labels

    def load_corpus(self, filetype):
        """
        filetype:
            train: load training data
            test : load testing data
            dev  : load development data
        """
        if filetype in ['train2', 'dev2', 'test2']:
            return self.__load_data(filetype)
        else:
            raise ValueError('mode error!')

    def load_lckg_embeddings(self):
        lckg = LCKG_Embeddings(self.lckg_dir, self.word2idxlckg)
        embeddings = lckg.load_pretrained_lckg()
        return embeddings

class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.dataset = data
        self.label = labels

    def __getitem__(self, index):
        data = self.dataset[index]
        label = self.label[index]
        return data, label

    def __len__(self):
        return len(self.label)


class MyDataLoader(object):
    def __init__(self, rel2id, config):
        self.rel2id = rel2id
        self.config = config
        self.corpus = MyCorpus(rel2id, config)

    def __collate_fn(self, batch):
        data, label = zip(*batch)  # unzip the batch data
        data = list(data)
        label = list(label)
        data = torch.from_numpy(np.concatenate(data, axis=0))
        label = torch.from_numpy(np.asarray(label, dtype=np.int64))
        return data, label

    def seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def __get_data(self, filetype, shuffle=False):
        data, labels = self.corpus.load_corpus(filetype)
        dataset = MyDataset(data, labels)
        g = torch.Generator()
        g.manual_seed(1234)
        loader = DataLoader(
            dataset=dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=2,
            collate_fn=self.__collate_fn,
            worker_init_fn=self.seed_worker,
            generator=g,
        )
        return loader

    def get_train(self):
        ret = self.__get_data(filetype='train2', shuffle=True)
        print('finish loading train!')
        return ret

    def get_dev(self):
        try:
            ret = self.__get_data(filetype='dev2', shuffle=False)
            print('finish loading dev!')
        except:
            ret = self.__get_data(filetype='test2', shuffle=False)
            print('finish loading test as dev!')
        return ret

    def get_test(self):
        ret = self.__get_data(filetype='test2', shuffle=False)
        print('finish loading test!')
        return ret

    def get_lckg(self):
        return self.corpus.load_lckg_embeddings()

if __name__ == '__main__':
    from config_x import Config
    config = Config()
    print('--------------------------------------')
    config.print_config()
    rel2id, id2rel, class_num = RelationLoader(config).get_relation()
    print (rel2id, id2rel, class_num)
    
    loader = MyDataLoader(rel2id, config)
    lckg_embeddings = torch.tensor(loader.get_lckg())
    print (lckg_embeddings.shape)
    test_loader = loader.get_test()

    # for step, (data, label) in enumerate(test_loader):
    #     print(type(data), data.shape)
    #     print(type(label), label.shape)
    #     import pdb
    #     pdb.set_trace()
    #     break

    train_loader = loader.get_train()
    dev_loader = loader.get_dev()
