from typing import BinaryIO
import numpy as np
from cs336_basics.train_utils import get_batch
import torch

class DataLoader:

    def __init__(self, datastet_path: str, batch_size, seq_len):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.dataset = np.memmap(datastet_path, np.uint16, 'r')
        self.num_sequences = len(self.dataset) - batch_size - seq_len - 1 # determine the number of unique sequences the data can be divided into
        self.epochs_completed = 0
        self._create_random_batch_order()
    
    def _create_random_batch_order(self):
        """Creates indices that span the start to the end of the indices"""
        indices = torch.arange(0, self.num_sequences, 32)
        np.random.shuffle(indices) # Randomly shuffle the order
        self.cur_seq = 0
        self.indices = indices

    def get_next_batch(self):
        if self.cur_seq == len(self.indices):
            self._create_random_batch_order()
        start_pos = self.indices[self.cur_seq]
        subset = self.dataset[start_pos:self.batch_size*self.seq_len+start_pos+1]
        x, y = get_batch(subset, self.batch_size, self.current_offset)
        self.cur_seq += 1
        return x, y

