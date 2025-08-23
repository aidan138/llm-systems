from cs336_basics.tokenizers.pretrained_tokenizer import PretrainedTokenizer
import numpy as np
import time
from cs336_basics.tokenizers.pretokenization_example import find_chunk_boundaries
import os

data_set_idx = 1

def main():
    tokenizer = PretrainedTokenizer.from_files(
        r"cs336_basics\tokenizers\trained_bpes\tiny_stories_vocab.pkl",
        r"cs336_basics\tokenizers\trained_bpes\tiny_stories_merges.pkl",
        ['<|endoftext|>']
    )

    datasets = [r'data\TinyStoriesV2-GPT4-train.txt', r'data\TinyStoriesV2-GPT4-valid.txt', r'data\owt_train.txt']
    dataset_path = datasets[data_set_idx]
    save_path = dataset_path.replace(".txt",'.npy')
    print(save_path)
    with open(dataset_path, 'rb') as f:
        chunks = find_chunk_boundaries(f, 1, '<|endoftext|>'.encode('utf-8'))
    
    print("Starting tokenization of document")
    start_time = time.perf_counter()
    tokenized_dataset = []

    with open(dataset_path, 'rb') as f:
        for start, end in zip(chunks[:-1], chunks[1:]):
            print(f'starting bytes {start}-{end} / {chunks[-1]} bytes')
            f.seek(start)
            txt = f.read(end-start).decode('utf-8', errors='ignore')
            tokens = tokenizer.encode(txt)
            tokenized_dataset.extend(tokens)

    tokenized_dataset = np.array(tokenized_dataset, np.uint16)
    np.save(save_path, tokenized_dataset)

    end_time = time.perf_counter()
    print(f"Finished tokenizing in {end_time-start_time} seconds")

if __name__ == '__main__':
    main()
