import torch


class Vocab:
    def __init__(self, text):
        self.text = text
        self.vocab = self.create_vocab()
        self.vocab_size = len(self.vocab)
        self.encode_dict = self.create_encode_dict()
        self.decode_dict = self.create_decode_dict()

    def create_vocab(self):
        unique_chars = set(self.text)
        return unique_chars

    def create_decode_dict(self):
        return {k: v for k, v in enumerate(self.vocab)}

    def create_encode_dict(self):
        return {v: k for k, v in enumerate(self.vocab)}


class TextEncoder:
    def __init__(self, vocab):
        self.vocab = vocab

    def encode(self, text):
        return [self.vocab.encode_dict[s] for s in text]

    def decode(self, numbers):
        return ''.join([self.vocab.decode_dict[n] for n in numbers])

    def convert_dataset_to_tensor(self, text):
        return torch.tensor(self.encode(text))
