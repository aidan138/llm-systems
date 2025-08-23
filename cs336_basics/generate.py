import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from cs336_basics.nn.utils import softmax
from cs336_basics.tokenizers.pretrained_tokenizer import PretrainedTokenizer
from cs336_basics.nn.layers import TransformerLM
from cs336_basics.train_utils import load_checkpoint
from cs336_basics.args import ModelArgs
from cs336_basics.nn.optim import AdamW


tiny_stories_files = [
        r"cs336_basics\tokenizers\trained_bpes\tiny_stories_vocab.pkl",
        r"cs336_basics\tokenizers\trained_bpes\tiny_stories_merges.pkl",
        ['<|endoftext|>']
    ]

model_checkpoint = r"models\first_run.pt"

model_args = ModelArgs(
        # LM config
        d_model=512,
        d_ff=1344,
        vocab_size=10000,
        rope_theta=10000,
        
        # Attention config
        num_layers=4,
        num_heads=16,

        # Inference time parameters
        max_batch_size=32,
        max_seq_len=256,
)



def decode(model: TransformerLM, tokenizer: PretrainedTokenizer, prompt: str, max_tokens: int, temperature: float = 0, p_scale = .90):

    model.eval()
    stop_seq = '<|endoftext|>'.encode('utf-8')
    with torch.no_grad():
        tokens = torch.Tensor(tokenizer.encode(prompt)).unsqueeze(0).int() # 1, N

        for _ in range(max_tokens):
            logits = model(tokens) # 1, N, V
            last_logits = logits[:, -1, :] # Only look at the last sequence logits
            if temperature == 0:
                next_token = torch.argmax(last_logits, dim=-1)
            else:
                tmp_scaled = last_logits / temperature
                pred_probs = softmax(tmp_scaled)
                top_p = []
                top_idx = []
                total_prob = 0

                while total_prob < p_scale:
                    curr_max = pred_probs.max(dim=-1) # 1
                    top_p.append(curr_max.values)
                    total_prob += curr_max.values
                    top_idx.append(curr_max.indices)
                    pred_probs[:, curr_max.indices] = float('-inf')
                
                idx = torch.multinomial(torch.Tensor([top_p]), num_samples=1)
                next_token = top_idx[idx]
                
            if tokenizer.vocab[next_token.item()] == stop_seq:
                break
            tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=-1)
            
    response = tokenizer.decode(tokens[0,:].tolist())
    return response
            


def main():
    lm = TransformerLM(model_args)
    optim = AdamW(lm.parameters())
    tokenizer = PretrainedTokenizer.from_files(*tiny_stories_files)
    load_checkpoint(model_checkpoint, lm, optim)
    output = decode(lm, tokenizer,"""Once upon a time""", 5000, .5)
    print(output)
    return

if __name__ == '__main__':
    tokenizer = PretrainedTokenizer.from_files(*tiny_stories_files)
    main()