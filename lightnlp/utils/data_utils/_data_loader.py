import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer
import numpy as np
import joblib
import zipfile
from functools import reduce
from sklearn.model_selection import train_test_split

from ..common import set_seed, ctqdm


class RawDataLoader:
    def __init__(self, verbose=1):
        self.verbose = verbose

    @staticmethod
    def _check_file_type(path):
        file_type = path.split(".")[-1]
        if file_type in ("txt", "csv", "tsv"):
            file_type = "txt"
        return file_type

    def _load_zip_to_list(self, path):
        ret = list()
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path, 'r') as f:
                for file in ctqdm(iterable=f.namelist(), verbose=self.verbose, desc="Load raw data"):
                    ret += f.read(file).split("\n")
        return ret

    def _load_txt_to_list(self, path):
        with open(path) as f:
            ret = [i.rstrip("\n") \
                   for i in ctqdm(iterable=f.readlines(), verbose=self.verbose, desc="Load raw data")]
        return ret

    def _load_to_list(self, path):
        file_type = self._check_file_type(path)
        if file_type == "zip":
            return self._load_zip_to_list(path)
        return self._load_txt_to_list(path)

    def _load_atc(self, path):
        corpus, label = list(), list()
        data = self._load_to_list(path)
        for d in data:
            l = d.strip().split("\t")
            corpus.append(l[0])
            label.append(l[1])
        return corpus, label

    def _load_ner(self, path):
        corpus, label = list(), list()
        data = self._load_to_list(path)
        for d in data:
            l = d.strip().split("_!_")
            corpus.append(l[0].strip().split(" "))
            label.append(l[1].strip().split(" "))
        return corpus, label

    def load_train(self, path, file_use):
        if file_use == "atc":
            return self._load_atc(path)
        if file_use == "ner":
            return self._load_ner(path)
        if file_use == "ats":
            return
        if file_use == "atg":
            return

    def load_predict(self, path):
        return self._load_to_list(path)


class EmbeddingLoader:
    """
    load pre-trained embedding data, vocab2idx, idx2vocab. \
        or parse the corpus and words.txt to generate embedding, vocab2idx, idx2vocab.
    the pre trained embedding data must be a word and vectors with space separating, like:
        你 0.245 0.442 0.012 ...
        我 0.244 0.413 0.057 ...
    """

    def __init__(self, sep=" ", dim=200, start="[SOS]", end="[END]", pad="[PAD]", unknown="[UNK]",
                 random_state=2020, verbose=1):
        self.sep = sep
        self.dim = dim
        self.start = start
        self.end = end
        self.pad = pad
        self.unknown = unknown
        self.random_state = random_state
        self.verbose = verbose

    def _load_pre_trained(self, path):
        from tqdm import tqdm
        embs = list()
        word_set = set()
        with open(path, encoding="utf-8") as f:
            for line in ctqdm(iterable=f.readlines(), verbose=self.verbose,
                              desc="Load pre-trained embed"):
                d = line.strip().split(self.sep)
                word, emb = d[0], [float(i) for i in d[1:]]
                word_set.add(word)
                if len(emb) < self.dim:
                    emb += [0.] * (self.dim - len(emb))
                embs.append([word] + emb)
        if self.pad not in word_set:
            embs.insert(0, [self.pad] + [0.] * self.dim)
        if self.start not in word_set:
            embs.insert(1, [self.start] + [0.] * self.dim)
        if self.end not in word_set:
            embs.insert(2, [self.end] + [0.] * self.dim)
        if self.unknown not in word_set:
            embs.insert(3, [self.unknown] + [0.] * self.dim)
        emb_weights = torch.tensor([e[1:] for e in embs])
        vocab2idx = {v[0]: i for i, v in enumerate(embs)}
        idx2vocab = {i: v[0] for i, v in enumerate(embs)}
        return emb_weights, vocab2idx, idx2vocab

    def _load_from_corpus(self, corpus, word_path=None):
        word_set = set()
        for c in corpus:
            for w in c.strip().replace(" ", ""):
                word_set.add(w)
        if word_path is not None:
            with open(word_path) as f:
                for line in f.readlines():
                    for w in line.strip():
                        word_set.add(w)
        set_seed(self.random_state)
        word_list = list(word_set)
        emb_weights = torch.randn((len(word_set), self.dim))
        if self.pad not in word_set:
            emb_weights = torch.cat((torch.zeros(1, self.dim), emb_weights), dim=0)
            word_list.insert(0, self.pad)
        if self.start not in word_set:
            emb_weights = torch.cat((torch.zeros(1, self.dim), emb_weights), dim=0)
            word_list.insert(1, self.start)
        if self.end not in word_set:
            emb_weights = torch.cat((torch.zeros(1, self.dim), emb_weights), dim=0)
            word_list.insert(2, self.end)
        if self.unknown not in word_set:
            emb_weights = torch.cat((torch.zeros(1, self.dim), emb_weights), dim=0)
            word_list.insert(3, self.unknown)
        vocab2idx = {v: i for i, v in enumerate(word_list)}
        idx2vocab = {i: v for i, v in enumerate(word_list)}
        return emb_weights, vocab2idx, idx2vocab

    def load(self, pre_trained_path=None, corpus=None, word_path=None):
        if pre_trained_path:
            return self._load_pre_trained(pre_trained_path)
        return self._load_from_corpus(corpus=corpus, word_path=word_path)


class BaseDataLoader:
    def __init__(self, train_size=None, batch_size=None,
                 vocab2idx=None, unknown_label="[UNK]", pad_label="[PAD]", max_length=None,
                 random_state=2020, verbose=1):
        self.train_size = train_size
        self.batch_size = batch_size
        self.vocab2idx = vocab2idx
        self.label2idx = None
        self.max_length = max_length
        self.unknown_label = unknown_label
        self.pad_label = pad_label
        self.random_state = random_state
        self.verbose = verbose
        self.label_num = None

    @staticmethod
    def _tokenize(sent,
                  label=None,
                  vocab2idx=None, label2idx=None,
                  unknown_label="[UNK]", pad_label="[PAD]",
                  max_length=None):

        sent_len = len(sent)
        sent_ids = list()
        for i in range(max_length):
            if i < sent_len:
                sent_ids.append(vocab2idx.get(sent[i], vocab2idx[unknown_label]))
            else:
                sent_ids.append(vocab2idx[pad_label])
        if label is None:
            return [sent_ids, None]

        label_index = label2idx[label]
        label_onehot = [1.0 if i == label_index else 0.0 for i in range(len(label2idx.keys()))]
        return [sent_ids, label_onehot]

    def load_train(self, corpus, label, n_jobs=1):
        corpus_max_length = min(max([len(c) for c in corpus]), 512)
        if self.max_length is None:
            self.max_length = corpus_max_length
        self.label2idx = {l: i for i, l in enumerate(np.unique(label))}
        self.label_num = len(self.label2idx.keys())

        data = joblib.Parallel(n_jobs)(
            joblib.delayed(self._tokenize) \
                (sent=s, label=l, max_length=self.max_length,
                 vocab2idx=self.vocab2idx, label2idx=self.label2idx,
                 unknown_label=self.unknown_label, pad_label=self.pad_label) \
            for s, l in ctqdm(iterable=zip(corpus, label), verbose=self.verbose,
                              desc="Tokenizing", total=len(corpus))
        )

        if self.train_size is None:
            tensor_dataset = TensorDataset(torch.tensor([d[0] for d in data]),
                                           torch.tensor([d[1] for d in data]))
            if self.batch_size is None:
                return tensor_dataset
            return DataLoader(dataset=tensor_dataset, batch_size=self.batch_size)
        else:
            train, test = train_test_split(data, train_size=self.train_size, random_state=self.random_state)

            train_tensor_dataset = TensorDataset(torch.tensor([d[0] for d in train]),
                                                 torch.tensor([d[1] for d in train]))
            test_tensor_dataset = TensorDataset(torch.tensor([d[0] for d in test]),
                                                torch.tensor([d[1] for d in test]))

            if self.batch_size is None:
                return [train_tensor_dataset, test_tensor_dataset]
            return [DataLoader(dataset=train_tensor_dataset, batch_size=self.batch_size),
                    DataLoader(dataset=test_tensor_dataset, batch_size=self.batch_size)]

    def load_predict(self, corpus, batch_size=None, n_jobs=1):
        data = joblib.Parallel(n_jobs)(
            joblib.delayed(self._tokenize) \
                (sent=s, label=None, max_length=self.max_length,
                 vocab2idx=self.vocab2idx, label2idx=self.label2idx,
                 unknown_label=self.unknown_label, pad_label=self.pad_label) \
            for s in ctqdm(iterable=corpus, desc="Tokenizing", verbose=self.verbose)
        )

        tensor_dataset = TensorDataset(torch.tensor([d[0] for d in data]))
        if batch_size is None:
            return tensor_dataset
        return DataLoader(dataset=tensor_dataset, batch_size=batch_size)


NERDataLoader = BaseDataLoader

ATCDataLoader = BaseDataLoader

ATGDataLoader = BaseDataLoader

ATSDataLoader = BaseDataLoader


class BertBaseDataLoader:
    def __init__(self, pre_trained_path=None, max_length=None,
                 batch_size=64, train_size=0.9,
                 random_state=2020, verbose=1, **kwargs):
        self.pre_trained_path = pre_trained_path
        self.tokenizer = BertTokenizer.from_pretrained(self.pre_trained_path)
        self.max_length = max_length
        self.batch_size = batch_size
        self.train_size = train_size
        self.random_state = random_state
        self.verbose = verbose

        self.label2idx = dict()
        self.label_num = None

    def _tokenize(self, s, l=None):
        encoded_dict = self.tokenizer.encode_plus(
            s,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=self.max_length,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt'  # Return pytorch tensors.
        )
        if l is None:
            return [encoded_dict['input_ids'], encoded_dict['attention_mask']]
        label_index = self.label2idx[l]
        label_ids = torch.tensor(
            [1.0 if i == label_index else 0.0 for i in range(len(self.label2idx.keys()))]
        ).reshape(1, -1)
        return [encoded_dict['input_ids'], label_ids, encoded_dict['attention_mask']]

    def load_train(self, corpus, label, n_jobs=1):
        self.label2idx = {l: i for i, l in enumerate(np.unique(label))}
        self.label_num = len(self.label2idx.keys())
        corpus_max_length = min(max([len(c) for c in corpus]), 512)
        if self.max_length is None:
            self.max_length = corpus_max_length

        data = joblib.Parallel(n_jobs)(
            joblib.delayed(self._tokenize)(c, l) \
            for c, l in ctqdm(iterable=zip(corpus, label),
                              total=len(corpus), desc="Tokenizing", verbose=self.verbose)
        )

        if self.train_size is None:
            tensor_dataset = TensorDataset(torch.cat([d[0] for d in data], dim=0),
                                           torch.cat([d[1] for d in data], dim=0),
                                           torch.cat([d[2] for d in data], dim=0))
            if self.batch_size is None:
                return tensor_dataset
            return DataLoader(dataset=tensor_dataset, batch_size=self.batch_size)
        else:
            train, test = train_test_split(data, train_size=self.train_size, random_state=self.random_state)
            train_tensor_dataset = TensorDataset(torch.cat([d[0] for d in train], dim=0),
                                                 torch.cat([d[1] for d in train], dim=0),
                                                 torch.cat([d[2] for d in train], dim=0),
                                                 )
            test_tensor_dataset = TensorDataset(torch.cat([d[0] for d in test], dim=0),
                                                torch.cat([d[1] for d in test], dim=0),
                                                torch.cat([d[2] for d in test], dim=0))

            if self.batch_size is None:
                return [train_tensor_dataset, test_tensor_dataset]
            return [DataLoader(dataset=train_tensor_dataset, batch_size=self.batch_size),
                    DataLoader(dataset=test_tensor_dataset, batch_size=self.batch_size)]

    def load_predict(self, corpus, batch_size=None, n_jobs=1):
        data = joblib.Parallel(n_jobs)(
            joblib.delayed(self._tokenize)(s=c) \
            for c in ctqdm(iterable=corpus, desc="tokenizing", verbose=self.verbose)
        )

        tensor_dataset = TensorDataset(torch.cat([d[0] for d in data], dim=0),
                                       torch.cat([d[1] for d in data], dim=0))
        if batch_size is None:
            return tensor_dataset
        return DataLoader(dataset=tensor_dataset, batch_size=batch_size)


BertBaseATCDataLoader = BertBaseDataLoader

AlbertBaseATCDataLoader = BertBaseDataLoader
