import copy
import json


with open('train.json', 'r', encoding='utf-8') as file:
    train = json.load(file)
with open('test.json', 'r', encoding='utf-8') as file:
    test = json.load(file)

import json
import os
import random
import collections
random.seed(42)
max_len=24
with open('hanzi.json', 'r', encoding='utf-8') as file:
    hanzis = json.load(file)
with open('source.json', 'r', encoding='utf-8') as file:
    source = json.load(file)

def count_corpus(tokens):  # @save
    """Count token frequencies."""
    # `tokens` can be a 1D list or a 2D list
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # Flatten a list of token lists into a single list
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)
class Vocab:
    """Vocabulary for text."""

    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        """Defined in :numref:`sec_text_preprocessing`"""
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # Sort according to frequencies
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        # The index for the unknown token is 0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # Index for the unknown token
        return 0

    @property
    def token_freqs(self):  # Index for the unknown token
        return self._token_freqs
src_vocab = Vocab(source, min_freq=0,
                      reserved_tokens=['<pad>', '<bos>','<sos>', '<eos>'])
def truncate_pad(line, num_steps, padding_token):
    """Truncate or pad a text sequence to a fixed length."""
    if len(line) > num_steps:
        return line[:num_steps]  # 截断
    return line + [padding_token] * (num_steps - len(line))  # 填充
def build_array_nmt(lines, vocab, num_steps):
    """Convert machine translation text sequences into mini-batches."""
    lines = [vocab[l] for l in lines]
    lines = [l + [vocab['<eos>']] for l in lines]
    array = [truncate_pad(
        l, num_steps, vocab['<pad>']) for l in lines]
    # valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
    valid_len=0
    return array, valid_len
with open('Deciphering_dataset/OBS_train.json', 'r', encoding='utf-8') as file:
    train_update = json.load(file)
print(len(train_update))
for root, directories, files in os.walk('../../Radical_Decomposition/output/results_train'):
    for file in files:
        file_path = os.path.join(root, file)
        folders = os.path.split(file_path)[0].split(os.sep)
        hanzi = folders[-1]
        # if hanzi  not in test:
        if True:
            xu, _ = build_array_nmt(hanzis[hanzi], src_vocab, max_len)
            for j in xu:
                data = {}
                inseq = [src_vocab['<sos>']] + j[:-1]
                data['input_seqs'] = inseq
                data['path'] = file_path
                data['label'] = hanzi
                data['output_seqs'] = j
                train_update.append(copy.deepcopy(data))
print(len(train_update))
with open('Deciphering_dataset/OBS_train_results_train.json','w',encoding='utf8') as f:
    json.dump(train_update, f, ensure_ascii=False)