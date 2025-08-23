from collections import defaultdict
from uuid import uuid4
from collections import Counter
import heapq
import regex as re
from cs336_basics.tokenizers.pretokenization_example import find_chunk_boundaries
from multiprocessing import cpu_count, Pool
import mmap
import time
import pickle
from pathlib import Path
import cProfile
import pstats

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Pretoken():
  def __init__(self, tokens, freq = 1):
    self.tokens = tokens # list of token ids
    self.freq = freq
    self.id = uuid4()

  def __hash__(self):
    return hash(self.id)

  def __eq__(self, x):
    if isinstance(x, Pretoken):
      return self.id == x.id
    raise NotImplementedError
  
  def _get_pairs(self):
    for pair in zip(self.tokens[:-1], self.tokens[1:]):
      yield pair

  def merge(self, merge_pair, new_token):
    tokens = []
    i = 0

    # Track the pairs modified and the number of times its modified
    prev_pair_counts = Counter(self._get_pairs())

    while i < len(self.tokens):
      if i + 1 < len(self.tokens) and (self.tokens[i], self.tokens[i+1]) == merge_pair:
        tokens.append(new_token)
        i+=2

      else:
        tokens.append(self.tokens[i])
        i+=1
    
    reduced_pairs = {}
    new_pairs = {}

    self.tokens = tuple(tokens)
    new_pair_counts = Counter(self._get_pairs())
    for pair in new_pair_counts | prev_pair_counts:
      diff = new_pair_counts.get(pair,0) - prev_pair_counts.get(pair, 0) if pair != merge_pair else 0
      if diff > 0:
        new_pairs[pair] = diff * self.freq
      elif diff < 0:
        reduced_pairs[pair] = -diff * self.freq

    return reduced_pairs, new_pairs

def get_stats(parent: Pretoken, stats: dict[tuple, int] = {}, bp_to_pt: dict[tuple, set] = {}):
    ids = parent.tokens
    increment = parent.freq
    for id_pair in zip(ids[:-1], ids[1:]):
        stats[id_pair] = stats.get(id_pair, 0) + increment # For sure works
        bp_to_pt[id_pair].add(parent)
    return stats


def count_chunk(args) -> Counter:
  input_path, start, end, spec_token_pattern = args
  with open(input_path, 'rb') as f, \
    mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mm:
    # Read the chunk from disk memory
    chunk = mm[start:end].decode('utf-8', errors='ignore')
  chunk = chunk.replace('\r\n', '\n').replace('\r', '\n')

  # Removes the last item if ends on a special token
  pretoken_counts = Counter()
  documents = re.split(spec_token_pattern, chunk)
  for document in documents:
    split_pretokens = re.finditer(PAT, document)
    pretoken_counts += Counter(tuple(pretoken.group().encode('utf-8')) for pretoken in split_pretokens)

  return pretoken_counts

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
  assert vocab_size >= (len(special_tokens) + 256)
  special_tok_pattern = '|'.join(re.escape(tok) for tok in special_tokens)
  special_start = vocab_size - len(special_tokens)
  special_token_dict = {special_start: special_tokens[i].encode('utf-8') for i in range(len(special_tokens))}
  start_time = time.perf_counter()

  start_cb_t = time.perf_counter()
  with open(input_path, 'rb') as f:
    num_processes = cpu_count()
    chunk_boundaries = find_chunk_boundaries(f, num_processes, special_tokens[0].encode('utf-8'))
  end_cb_t = time.perf_counter()

  print(f"Finished getting chunk boundaries in {end_cb_t-start_cb_t}")

  start_pt_t = time.perf_counter()
  pretoken_counts = Counter()
  arguments = [
      (input_path, chunk_boundaries[i], chunk_boundaries[i+1], special_tok_pattern) for i in range(len(chunk_boundaries)-1)
    ]
  
  with Pool() as pool:
    results = pool.map(count_chunk, arguments)
  
  for result in results:
    pretoken_counts.update(result)
  end_pt_t = time.perf_counter()
  print(f"Finished counting chunks in {end_pt_t-start_pt_t}")

  start_tok_t = time.perf_counter()
  pretokens = [Pretoken(tok, freq) for tok, freq in pretoken_counts.items()]
  byte_freqs = {} # byte_pair : freq
  bp_to_pt = defaultdict(set)# byte_pair : set(parent Pretokens)

  for pretoken in pretokens:
    get_stats(pretoken, byte_freqs, bp_to_pt)

  heap = [(-freq, tuple(-x for x in bp)) for bp, freq in byte_freqs.items()] # Inverse values for min priority queue
  heapq.heapify(heap)  

  # Non parrallelizable merging
  num_merges = vocab_size - len(special_tokens) - 256
  current_token = 256

  # Establish return values
  vocab = {i: bytes([i]) for i in range(256)} | special_token_dict  # Token: bytes representation
  merges = {} # (byte, byte) -> merge_byte

  # Main merge loop
  for _ in range(num_merges):
    while heap:
      # TODO Optimize the tie breaks by refactoring datastructures to use bytes

      # Tie break: Choose the token pair whose bytes are lexicographically largest
      neg_freq0, neg_pair0 = heapq.heappop(heap)

      # Gather all the others with the same neg_freq
      same_freq = [(neg_freq0, neg_pair0)]
      while heap and heap[0][0] == neg_freq0:
          same_freq.append(heapq.heappop(heap))

      # Tie break sequence
      best = max(
          same_freq,
          key=lambda freq_pair: (
              vocab[-freq_pair[1][0]],
              vocab[-freq_pair[1][1]]
          )
      )

      # Push extraneous same frequencies back
      for item in same_freq:
          if item is not best:
              heapq.heappush(heap, item)

      neg_freq, neg_pair = best
      pair = tuple(-x for x in neg_pair)

      # Check for a valid merge
      if neg_freq != 0 and pair in byte_freqs and -neg_freq == byte_freqs[pair]:
        vocab[current_token] = vocab[pair[0]] + vocab[pair[1]]
        merges[pair] = current_token

        byte_freqs.pop(pair)

        parents = bp_to_pt.pop(pair)

        for parent in parents:
          reduced_pairs, new_pairs = parent.merge(pair, current_token)

          for mod_pair, mod_freq in reduced_pairs.items():
            byte_freqs[mod_pair] -= mod_freq
            heapq.heappush(heap, (-byte_freqs[mod_pair], tuple(-x for x in mod_pair)))

          for mod_pair, mod_freq in new_pairs.items():
            byte_freqs[mod_pair] = byte_freqs.get(mod_pair, 0) + mod_freq
            bp_to_pt[mod_pair].add(parent)
            heapq.heappush(heap, (-byte_freqs[mod_pair], tuple(-x for x in mod_pair)))

        current_token += 1
        break
  end_time = time.perf_counter()
  print(f"Finished merging in {end_time - start_tok_t}")
  print(f"Full tokenization time {end_time - start_time}")
  
  # The more helpful merge representation, not used to match industry standard
  # Could be used in later implementations
  real_merges = merges 

  merges = [(vocab[key[0]], vocab[key[1]]) for key in merges.keys()]
    
  return vocab, merges

if __name__ == '__main__':

  profiler = cProfile.Profile()
  profiler.enable()
  vocab, merges = train_bpe(r'data\TinyStoriesV2-GPT4-train.txt', 10000, ["<|endoftext|>"])
  profiler.disable()
  stats = pstats.Stats(profiler)
  profiler.dump_stats("train_bpe_profile.prof")
  stats.sort_stats("cumtime").print_stats(20)  # top 20 cumulative time functions

  bpe_name = input("Input the bpe name: ")
  file_dir = 'trained_bpes'
  current_dir = Path(__file__).resolve().parent
  file_dir_path = current_dir / file_dir

  file_dir_path.mkdir(exist_ok=True)
  
  filename = f"{bpe_name}_vocab.pkl"
  filepath = file_dir_path / filename

  with open(filepath, 'wb') as f:
    pickle.dump(vocab, f)
  
  filename = f"{bpe_name}_merges.pkl"
  filepath = file_dir_path / filename
  with open(filepath, 'wb') as f:
    pickle.dump(merges, f)
  
  # Print out relevant stats
  print(f"The longest token is: {max(vocab.values(), key= lambda v: len(v))}")
  