from abc import ABC, abstractmethod


class Tokenizer(ABC):

    def __init__(self):
        self.merges = {} # (token, token) -> token
        self.vocab = {i: bytes([i]) for i in range(256)} # int to bytes
        self.current_token = 256

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def encode(self):
        pass

    @abstractmethod
    def decode(self):
        pass
