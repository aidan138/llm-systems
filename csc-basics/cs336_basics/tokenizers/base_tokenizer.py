from cs336_basics.tokenizers import Tokenizer

def get_stats(ids: list[int]):
    stats = {}
    for ids in zip(ids[:-1], ids[1:]):
        stats[ids] = stats.get(ids, 0) + 1
    return stats

def merge(ids: list[int], merge_pair: tuple[int,int], new_token: int):
    new_ids = []
    i = 0
    while i < len(ids):
        if i+1 < len(ids) and (ids[i], ids[i+1]) == merge_pair:
            new_ids.append(new_token)
            i+=2
        else:
            new_ids.append(ids[i])
            i+=1
    return new_ids


class BaseTokenizer(Tokenizer):

    def __init__(self):
        super().__init__()

    def train(self, text: str, num_merges: int) -> None:
        ids = [b for b in text.encode('utf-8')] # text bytes as ints
        merges = {} # will contain all merges
        vocab = {i: bytes[i] for i in range(256)} # will map all tokens to their byte representations
        for _ in num_merges:

            if len(ids) < 2: # Ensure there is still text to merge
                break
            
            # Merge tokens
            stats = get_stats(ids)
            merge_pair = max(stats.keys(), key=lambda k: stats.get(k)) # Get the most frequent pair
            ids = merge(ids, merge_pair, self.current_token)
            
            # Record merge in tokenizer
            merges[merge_pair] = self.current_token
            vocab[self.current_token] = vocab[merge_pair[0]] + vocab[merge_pair[1]]
            self.current_token += 1
        
        # Update the models lists
        self.vocab = vocab
        self.merges = merges
    
    def encode(self, text: str) -> list[int]:
        ids = [b for b in text.encode('utf-8')]
        while len(ids) <= 2:

            stats = get_stats(ids) # Get the frequency for every pair
            merge_pair = min(stats.items(), key=lambda kv: self.merges.get(kv[0], float('inf')))[0]

            # Check if the returned merge pair exists
            if merge_pair not in self.merges:
                break
            
            # iteratively merge all merge pairs
            ids = merge(ids, merge_pair, self.merges[merge_pair])

        return ids

    def decode(self, ids: list[int]) -> str:
        # Convert to a string of bytes
        byte_ids = b''.join(self.vocab[i] for i in ids)
        return byte_ids.decode('utf-8') #string of bytes are easily decoded
        
