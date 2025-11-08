from dataclasses import dataclass
from typing import List, Optional

import pytorch_lightning as pl
import torch
from more_itertools import transpose
from torch import FloatTensor, LongTensor
from torch.utils.data.dataloader import DataLoader

from ical.datamodule.dataset import CROHMEDataset

from .vocab import vocab


@dataclass
class Batch:
    img_bases: List[str]  # [b,]
    imgs: FloatTensor  # [b, 1, H, W]
    mask: LongTensor  # [b, H, W]
    indices: List[List[int]]  # [b, l]

    def __len__(self) -> int:
        return len(self.img_bases)

    def to(self, device) -> "Batch":
        return Batch(
            img_bases=self.img_bases,
            imgs=self.imgs.to(device),
            mask=self.mask.to(device),
            indices=self.indices,
        )


def collate_fn(batch):
    fnames, images_x, captions_y = transpose(batch)

    seqs_y = [vocab.words2indices(x) for x in captions_y]

    heights_x = [s.size(1) for s in images_x]
    widths_x = [s.size(2) for s in images_x]

    n_samples = len(heights_x)
    max_height_x = max(heights_x)
    max_width_x = max(widths_x)

    x = torch.zeros(n_samples, 3, max_height_x, max_width_x)
    x_mask = torch.ones(n_samples, max_height_x, max_width_x, dtype=torch.bool)
    for idx, s_x in enumerate(images_x):
        x[idx, :, : heights_x[idx], : widths_x[idx]] = s_x
        x_mask[idx, : heights_x[idx], : widths_x[idx]] = 0

    # return fnames, x, x_mask, seqs_y
    return Batch(fnames, x, x_mask, seqs_y)


class CROHMEDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        path: str,
        test_split: str,
        w: int,
        h: int,
        fill: int,
        line: int,
        train_batch_size: int,
        eval_batch_size: int,
        num_workers: int,
    ) -> None:
        super().__init__()

        self.path = path
        self.test_split = test_split
        self.w = w
        self.h = h
        self.fill = fill
        self.line = line

        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers

        print(f"Load data from: {self.path}")

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = CROHMEDataset(
                path=self.path,
                split="train",
                w=self.w,
                h=self.h,
                fill=self.fill,
                line=self.line,
            )
            self.val_dataset = CROHMEDataset(
                path=self.path,
                split="test",
                w=self.w,
                h=self.h,
                fill=self.fill,
                line=self.line,
            )
        if stage in ["test", "predict"] or stage is None:
            self.test_dataset = CROHMEDataset(
                path=self.path,
                split=self.test_split,
                w=self.w,
                h=self.h,
                fill=self.fill,
                line=self.line,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def predict_dataloader(self):
        return self.test_dataloader()
