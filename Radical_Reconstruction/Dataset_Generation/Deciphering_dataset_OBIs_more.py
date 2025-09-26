import argparse
import collections
import copy
import json
import os
import random


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Augment OBS train dataset with additional OCR results."
    )
    parser.add_argument(
        "--train",
        default="train.json",
        help="Path to the train.json split file."
    )
    parser.add_argument(
        "--test",
        default="test.json",
        help="Path to the test.json split file."
    )
    parser.add_argument(
        "--hanzi",
        default="../data/hanzi.json",
        help="Path to the hanzi.json corpus file."
    )
    parser.add_argument(
        "--source",
        default="../data/source.json",
        help="Path to the source.json corpus file."
    )
    parser.add_argument(
        "--obs-train",
        default="Deciphering_dataset/OBS_train.json",
        dest="obs_train",
        help="Existing OBS train dataset to be extended."
    )
    parser.add_argument(
        "--results-root",
        default="../../Radical_Decomposition/output/results_train",
        help="Root directory containing OCR results to ingest."
    )
    parser.add_argument(
        "--output",
        default="Deciphering_dataset/OBS_train_results_train.json",
        help="Destination path for the augmented OBS train dataset."
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=24,
        dest="max_len",
        help="Maximum sequence length for truncation and padding."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for reproducible shuffling if needed."
    )
    return parser.parse_args()


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


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
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
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
    def token_freqs(self):
        return self._token_freqs


def truncate_pad(line, num_steps, padding_token):
    """Truncate or pad a text sequence to a fixed length."""
    if len(line) > num_steps:
        return line[:num_steps]
    return line + [padding_token] * (num_steps - len(line))


def build_array_nmt(lines, vocab, num_steps):
    """Convert machine translation text sequences into sequences of ids."""
    lines = [vocab[l] for l in lines]
    lines = [l + [vocab['<eos>']] for l in lines]
    array = [truncate_pad(l, num_steps, vocab['<pad>']) for l in lines]
    # valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
    valid_len = 0
    return array, valid_len


def dump_json(path: str, payload):
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    _ = load_json(args.train)
    _ = load_json(args.test)
    hanzis = load_json(args.hanzi)
    source = load_json(args.source)
    train_update = load_json(args.obs_train)

    print(len(train_update))

    src_vocab = Vocab(
        source,
        min_freq=0,
        reserved_tokens=['<pad>', '<bos>', '<sos>', '<eos>']
    )

    for root, _, files in os.walk(args.results_root):
        for file in files:
            file_path = os.path.join(root, file)
            folders = os.path.split(file_path)[0].split(os.sep)
            hanzi = folders[-1]
            sequences, _ = build_array_nmt(hanzis[hanzi], src_vocab, args.max_len)
            # sequences=[sequences[0]] # or only use one sequence
            for sequence in sequences:
                data = {
                    'input_seqs': [src_vocab['<sos>']] + sequence[:-1],
                    'path': file_path,
                    'label': hanzi,
                    'output_seqs': sequence,
                }
                train_update.append(copy.deepcopy(data))

    print(len(train_update))
    dump_json(args.output, train_update)


if __name__ == "__main__":
    main()
