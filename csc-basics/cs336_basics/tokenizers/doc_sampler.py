from cs336_basics.tokenizers.pretrained_tokenizer import PretrainedTokenizer
import regex as re

def main():
    tokenizer = PretrainedTokenizer.from_files(
        r"cs336_basics\tokenizers\trained_bpes\tiny_stories_vocab.pkl",
        r"cs336_basics\tokenizers\trained_bpes\tiny_stories_merges.pkl",
        ['<|endoftext|>']
    )

    datasets = [r'data\TinyStoriesV2-GPT4-train.txt', r'data\owt_train.txt']
    compression_ratios = {}
    for dataset_path in datasets:
        ratio_list = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            txt = f.read(1000000) # Read the 1st MB
            documents = re.split(tokenizer.spec_pat, txt)
            for i in range(10):
                i = min(i, len(documents)-1)
                document = documents[i]
                #print(document)
                compression_ratio = ((1 - len(tokenizer.encode(document)) / len(list([b for b in document.encode('utf-8')]))) * 100)
                print(f"Document {i} has compression ratio: {compression_ratio:.3f}%")
                ratio_list.append(compression_ratio)
        compression_ratios[dataset_path] = sum(ratio_list) / len(ratio_list)
        print(f"Dataset: {dataset_path} had a mean compression ratio of {compression_ratios[dataset_path]:.3f}%")
    relative_compression = compression_ratios[datasets[0]] / compression_ratios[datasets[1]]
    print(f"The tiny stories : OWT compression ratio = {relative_compression:.3f}")
    

if __name__ == '__main__':
    main()
