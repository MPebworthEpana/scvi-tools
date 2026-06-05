"""Sortish batch sampler that groups cells of similar ATAC length."""

from __future__ import annotations

import numpy as np
from torch.utils.data import Sampler


class LengthBucketedBatchSampler(Sampler):
    """Group cells of similar ATAC length into batches.

    Each epoch: shuffle -> chunk into mega-buckets (``bucket_mult`` x batch) -> sort each
    bucket by length -> split into batches -> shuffle batch order. Yields lists of dataset
    positions (0..N-1). Reseeds per ``__iter__`` (Lightning re-iterates per epoch).
    """

    def __init__(
        self,
        lengths,
        batch_size,
        *,
        shuffle=True,
        drop_last=False,
        seed=0,
        bucket_mult=50,
    ):
        self.lengths = np.asarray(lengths)
        self.bs = int(batch_size)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = int(seed)
        self.bucket = max(1, int(bucket_mult)) * self.bs
        self._epoch = 0

    def _num_batches(self) -> int:
        n = len(self.lengths)
        total = 0
        for i in range(0, n, self.bucket):
            c = min(self.bucket, n - i)
            total += c // self.bs if self.drop_last else (c + self.bs - 1) // self.bs
        return total

    def __len__(self) -> int:
        return self._num_batches()

    def __iter__(self):
        n = len(self.lengths)
        idx = np.arange(n)
        rng = np.random.default_rng(self.seed + self._epoch)
        self._epoch += 1
        if self.shuffle:
            rng.shuffle(idx)
        batches = []
        for i in range(0, n, self.bucket):
            chunk = idx[i : i + self.bucket]
            chunk = chunk[np.argsort(self.lengths[chunk], kind="stable")]
            for j in range(0, len(chunk), self.bs):
                b = chunk[j : j + self.bs]
                if self.drop_last and len(b) < self.bs:
                    continue
                batches.append(b.tolist())
        if self.shuffle:
            batches = [batches[k] for k in rng.permutation(len(batches))]
        return iter(batches)
