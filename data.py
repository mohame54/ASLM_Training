import numpy as np
import torch
import os
from utils import load_json


class DataLoaderLite:
    def __init__(
        self,
        data_root,
        batch_size,
        seq_len,
        process_rank,
        num_processes,
        split
    ):
        self.B = batch_size
        self.T = seq_len
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        
        data_root = data_root
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s and s.endswith(".npy")]
        self.lengths_config = load_json(os.path.join(data_root, [s for s in os.listdir(data_root) if s.endswith(".json")][0]))
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if process_rank == 0:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def load_tokens(self, file_path):
        filename = file_path.split("/"[-1])
        npt = np.load(file_path)
        l = self.lengths_config[filename]
        npt = np.ascontiguousarray(npt.astype(np.int32)[:l])
        ptt = torch.tensor(npt, dtype=torch.long)
        return ptt    

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = self.load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = self.load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y
