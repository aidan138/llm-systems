from cs336_basics.tokenizers.tokenizer import Tokenizer
# from cs336_basics.pretokenization_example import find_text_boundaries
import json
from typing import Iterable, Iterator
import regex as re
import pickle


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def merge_pretoken(pretoken: list[bytes], merge_pair: tuple[bytes, bytes]) -> list[bytes]:
    i = 0
    new_pretoken = []
    merge = merge_pair[0] + merge_pair[1]

    while i < len(pretoken):
        if i + 1 < len(pretoken) and (pretoken[i], pretoken[i+1]) == merge_pair:
            new_pretoken.append(merge)
            i += 2
        
        else:
            new_pretoken.append(pretoken[i])
            i += 1
    return new_pretoken



class PretrainedTokenizer(Tokenizer):

    def __init__ (self, vocab: dict[int: bytes], merges: list[tuple[bytes,bytes]], special_tokens: list[str] | None = None, pat = None):    
        special_tokens = None if special_tokens is None else sorted(special_tokens, key=len, reverse=True)
        self.vocab = vocab
        self.merges = merges

        self.spec_tok_cache = {} if special_tokens is None else self._build_special_token_cache(special_tokens) # [str, int]
        self.reverse_vocab = {bt: token for token, bt in vocab.items()} # For quick encoding
        self.pat = PAT if pat is None else pat
        self.spec_pat = '|'.join(re.escape(tok) for tok in special_tokens) if special_tokens else None


    def _build_special_token_cache(self, special_tokens: list[str] = None):
        assert(self.vocab.get(len(self.vocab)) is None) # Ensures vocab is not supposed to be larger than it is
        spec_to_tok = {}
        current_token = len(self.vocab)
        bytes_set = set(self.vocab.values())
        for special_token in special_tokens:
            byte_token = special_token.encode('utf-8')
            if byte_token not in bytes_set:
                spec_to_tok[special_token] = current_token 
                self.vocab[current_token] = byte_token
                current_token += 1
            else:
                # If it is in the vocab cache its true token index
                token_int = next(filter(lambda kv: kv[1] == byte_token, self.vocab.items()))[0]
                spec_to_tok[special_token] = token_int

        return spec_to_tok
        

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):

        if vocab_filepath.endswith('.json'):
            with open(vocab_filepath, 'r') as f:
                raw_vocab = json.load(f)
                vocab = {token: str_byte.encode('utf-8') for str_byte, token in raw_vocab.items()}

        elif vocab_filepath.endswith('.pkl'):
            with open(vocab_filepath, 'rb') as f:
                vocab = pickle.load(f)
        else:
            raise NotImplementedError

        if merges_filepath.endswith('.txt'):
            merges = []
            with open(merges_filepath, 'r', encoding="utf-8") as f:
                lines = f.readlines()
                for line in lines:
                    pair = line.strip().split(' ')
                    tok1, tok2 = pair[0], pair[1]
                    merges.append(tuple((tok1.encode('utf-8'), tok2.encode('utf-8'))))
        elif merges_filepath.endswith('.pkl'):
            with open(merges_filepath, 'rb') as f:
                merges = pickle.load(f)
        
        else:
            raise NotImplementedError

            
        return PretrainedTokenizer(vocab, merges, special_tokens)


    def encode_pretoken(self, pretoken_bytes: list[bytes]):
        
        while len(pretoken_bytes) > 1:
            byte_pairs = set(zip(pretoken_bytes[:-1], pretoken_bytes[1:]))
            merge_pair = next(filter(lambda merge: merge in byte_pairs, self.merges), None)
            
            # No possible merges means end early
            if not merge_pair:
                break
            pretoken_bytes = merge_pretoken(pretoken_bytes, merge_pair)

        return list(map(lambda b: self.reverse_vocab[b], pretoken_bytes))

    def yield_special_txt(self, text):
        prev_end = 0
        split_text = []
        for special_sequence in re.finditer(self.spec_pat, text):
            start, end = special_sequence.span()
            if start > prev_end:
                yield text[prev_end:start]
            yield text[start:end]
            prev_end = end
        
        if prev_end < len(text):
            yield text[prev_end:]
    
    def encode(self, text: str) -> list[int]:
        # Pretokenize the groups
        special_split_text = self.yield_special_txt(text) if self.spec_pat else [text]
        encoded_ids = []

        for chunk in special_split_text:
            # Check for special tokens
            if chunk in self.spec_tok_cache:
                encoded_ids.append(self.spec_tok_cache[chunk])
            else:
                pretoken_groups = re.finditer(self.pat, chunk)
                for pretoken_group in pretoken_groups:
                    # merge pretoken bytes and cast into the integer 
                    pretoken_bytes = [bytes([b]) for b in pretoken_group.group().encode('utf-8')]
                    encoded_ids.extend(self.encode_pretoken(pretoken_bytes))

        return encoded_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Output is an iterator that lazily returns token ids"""
        for text in iterable:
            for id in self.encode(text):
                yield id


    def decode(self, ids: list[int]) -> str:
        byte_txt = b''.join(self.vocab[id] if id in self.vocab else b'\xEF\xBF\xBD' for id in ids)
        return byte_txt.decode('utf-8', errors='replace')

    def train(self):
        raise NotImplementedError

if __name__ == '__main__':
    tokenizer = PretrainedTokenizer.from_files(r'tests\fixtures\gpt2_vocab.json', r'tests\fixtures\gpt2_merges.txt', ['<|endoftext|>'])
    