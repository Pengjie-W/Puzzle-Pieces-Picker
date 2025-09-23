import copy
import json
import os
import random
import collections
random.seed(42)
with open('hanzi.json', 'r', encoding='utf-8') as file:
    hanzis = json.load(file)
with open('source.json', 'r', encoding='utf-8') as file:
    source = json.load(file)
with open('../data/ID_to_Chinese.json', 'r', encoding='utf-8') as file:
    ID_to_Chinese = json.load(file)
max_len=24
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
OBS=[]
for root, directories, files in os.walk('../data/Dataset'):
    for file in files:
        file_path = os.path.join(root, file)
        folders = os.path.split(file_path)[0].split(os.sep)
        hanzi = folders[-1]
        if file[0] == 'O':
            if hanzi not in OBS:
                OBS.append(hanzi)
            continue
random.shuffle(OBS)
# train=OBS[:-1000]
# test=OBS[-1000:]
with open('train.json', 'r', encoding='utf-8') as file:
    train = json.load(file)
with open('test.json', 'r', encoding='utf-8') as file:
    test = json.load(file)
OBS_train=[]
OBS_test=[]
Bronze_train=[]
Bronze_test=[]
Warring_train=[]
Warring_test=[]
Seal_train=[]
Seal_test=[]
Clerical_train=[]
Clerical_test=[]
Kangxi_train=[]
Kangxi_test=[]
Regular_train=[]
Regular_test=[]
for root, directories, files in os.walk('../data/dataset'):
    for file in files:
        file_path = os.path.join(root, file)
        folders = os.path.split(file_path)[0].split(os.sep)
        hanzi = ID_to_Chinese[folders[-1]]
        xu, _ = build_array_nmt(hanzis[hanzi], src_vocab, max_len)
        for j in xu:
            data = {}
            inseq = [src_vocab['<sos>']] + j[:-1]
            data['input_seqs'] = inseq
            data['path'] = file_path
            data['label'] = hanzi
            data['output_seqs'] = j
            if hanzi in test:
                if file[0] == 'O':
                    OBS_test.append(copy.deepcopy(data))
                    Bronze_train.append(copy.deepcopy(data))
                    Warring_train.append(copy.deepcopy(data))
                    Seal_train.append(copy.deepcopy(data))
                    Clerical_train.append(copy.deepcopy(data))
                    Kangxi_train.append(copy.deepcopy(data))
                    Regular_train.append(copy.deepcopy(data))
                elif file[0] == 'J':
                    OBS_train.append(copy.deepcopy(data))
                    Bronze_test.append(copy.deepcopy(data))
                    Warring_train.append(copy.deepcopy(data))
                    Seal_train.append(copy.deepcopy(data))
                    Clerical_train.append(copy.deepcopy(data))
                    Kangxi_train.append(copy.deepcopy(data))
                    Regular_train.append(copy.deepcopy(data))
                elif file[0] == 'W':
                    OBS_train.append(copy.deepcopy(data))
                    Bronze_train.append(copy.deepcopy(data))
                    Warring_test.append(copy.deepcopy(data))
                    Seal_train.append(copy.deepcopy(data))
                    Clerical_train.append(copy.deepcopy(data))
                    Kangxi_train.append(copy.deepcopy(data))
                    Regular_train.append(copy.deepcopy(data))
                elif file[0] == 'Z':
                    OBS_train.append(copy.deepcopy(data))
                    Bronze_train.append(copy.deepcopy(data))
                    Warring_train.append(copy.deepcopy(data))
                    Seal_test.append(copy.deepcopy(data))
                    Clerical_train.append(copy.deepcopy(data))
                    Kangxi_train.append(copy.deepcopy(data))
                    Regular_train.append(copy.deepcopy(data))
                elif file[0] == 'L':
                    OBS_train.append(copy.deepcopy(data))
                    Bronze_train.append(copy.deepcopy(data))
                    Warring_train.append(copy.deepcopy(data))
                    Seal_train.append(copy.deepcopy(data))
                    Clerical_test.append(copy.deepcopy(data))
                    Kangxi_train.append(copy.deepcopy(data))
                    Regular_train.append(copy.deepcopy(data))
                elif file[0] == 'X':
                    OBS_train.append(copy.deepcopy(data))
                    Bronze_train.append(copy.deepcopy(data))
                    Warring_train.append(copy.deepcopy(data))
                    Seal_train.append(copy.deepcopy(data))
                    Clerical_train.append(copy.deepcopy(data))
                    Kangxi_test.append(copy.deepcopy(data))
                    Regular_train.append(copy.deepcopy(data))
                elif file[0] == 'K':
                    OBS_train.append(copy.deepcopy(data))
                    Bronze_train.append(copy.deepcopy(data))
                    Warring_train.append(copy.deepcopy(data))
                    Seal_train.append(copy.deepcopy(data))
                    Clerical_train.append(copy.deepcopy(data))
                    Kangxi_train.append(copy.deepcopy(data))
                    Regular_test.append(copy.deepcopy(data))
                else:
                    print(file_path)
            else:
                OBS_train.append(copy.deepcopy(data))
                Bronze_train.append(copy.deepcopy(data))
                Warring_train.append(copy.deepcopy(data))
                Seal_train.append(copy.deepcopy(data))
                Clerical_train.append(copy.deepcopy(data))
                Kangxi_train.append(copy.deepcopy(data))
                Regular_train.append(copy.deepcopy(data))

for root, directories, files in os.walk('../data/Font_Generation'):
    for file in files:
        file_path = os.path.join(root, file)
        folders = os.path.split(file_path)[0].split(os.sep)
        name_without_ext = os.path.splitext(file)[0]
        hanzi = ID_to_Chinese[name_without_ext]
        xu, _ = build_array_nmt(hanzis[hanzi], src_vocab, max_len)
        for j in xu:
            data = {}
            inseq = [src_vocab['<sos>']] + j[:-1]
            data['input_seqs'] = inseq
            data['path'] = file_path
            data['label'] = hanzi
            data['output_seqs'] = j
            if hanzi in test:
                OBS_train.append(copy.deepcopy(data))
                Bronze_train.append(copy.deepcopy(data))
                Warring_train.append(copy.deepcopy(data))
                Seal_train.append(copy.deepcopy(data))
                Clerical_train.append(copy.deepcopy(data))
                Kangxi_train.append(copy.deepcopy(data))
                Regular_test.append(copy.deepcopy(data))
            else:
                OBS_train.append(copy.deepcopy(data))
                Bronze_train.append(copy.deepcopy(data))
                Warring_train.append(copy.deepcopy(data))
                Seal_train.append(copy.deepcopy(data))
                Clerical_train.append(copy.deepcopy(data))
                Kangxi_train.append(copy.deepcopy(data))
                Regular_train.append(copy.deepcopy(data))
if not os.path.exists('Deciphering_dataset'):
        os.makedirs('Deciphering_dataset')
with open('Deciphering_dataset/OBS_train.json','w',encoding='utf8') as f:
    json.dump(OBS_train, f, ensure_ascii=False)
print('OBS_train',len(OBS_train))

with open('Deciphering_dataset/OBS_test.json','w',encoding='utf8') as f:
    json.dump(OBS_test, f, ensure_ascii=False)
print('OBS_test',len(OBS_test))

with open('Deciphering_dataset/Bronze_train.json','w',encoding='utf8') as f:
    json.dump(Bronze_train, f, ensure_ascii=False)
print('Bronze_train',len(Bronze_train))

with open('Deciphering_dataset/Bronze_test.json','w',encoding='utf8') as f:
    json.dump(Bronze_test, f, ensure_ascii=False)
print('Bronze_test',len(Bronze_test))

with open('Deciphering_dataset/Warring_train.json','w',encoding='utf8') as f:
    json.dump(Warring_train, f, ensure_ascii=False)
print('Warring_train',len(Warring_train))

with open('Deciphering_dataset/Warring_test.json','w',encoding='utf8') as f:
    json.dump(Warring_test, f, ensure_ascii=False)
print('Warring_test',len(Warring_test))

with open('Deciphering_dataset/Seal_train.json','w',encoding='utf8') as f:
    json.dump(Seal_train, f, ensure_ascii=False)
print('Seal_train',len(Seal_train))

with open('Deciphering_dataset/Seal_test.json','w',encoding='utf8') as f:
    json.dump(Seal_test, f, ensure_ascii=False)
print('Seal_test',len(Seal_test))

with open('Deciphering_dataset/Clerical_train.json','w',encoding='utf8') as f:
    json.dump(Clerical_train, f, ensure_ascii=False)
print('Clerical_train',len(Clerical_train))

with open('Deciphering_dataset/Clerical_test.json','w',encoding='utf8') as f:
    json.dump(Clerical_test, f, ensure_ascii=False)
print('Clerical_test',len(Clerical_test))

with open('Deciphering_dataset/Kangxi_train.json','w',encoding='utf8') as f:
    json.dump(Kangxi_train, f, ensure_ascii=False)
print('Kangxi_train',len(Kangxi_train))

with open('Deciphering_dataset/Kangxi_test.json','w',encoding='utf8') as f:
    json.dump(Kangxi_test, f, ensure_ascii=False)
print('Kangxi_test',len(Kangxi_test))

with open('Deciphering_dataset/Regular_train.json','w',encoding='utf8') as f:
    json.dump(Regular_train, f, ensure_ascii=False)
print('Regular_train',len(Regular_train))

with open('Deciphering_dataset/Regular_test.json','w',encoding='utf8') as f:
    json.dump(Regular_test, f, ensure_ascii=False)
print('Regular_test',len(Regular_test))

with open('Deciphering_dataset/train.json','w',encoding='utf8') as f:
    json.dump(train, f, ensure_ascii=False)
with open('Deciphering_dataset/test.json','w',encoding='utf8') as f:
    json.dump(test, f, ensure_ascii=False)
