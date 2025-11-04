from time import perf_counter_ns
from typing import List

import pytorch_lightning as pl
import torch
import torch.optim as optim
from torch import FloatTensor, LongTensor

from ical.datamodule import Batch, vocab
from ical.model.ical import ICAL
from ical.utils.utils import (
    CERRecorder,
    ExpRateRecorder,
    FPSRecorder,
    Hypothesis,
    ce_loss,
    plicit_tgt_out,
)


class LitICAL(pl.LightningModule):
    def __init__(
        self,
        d_model: int,
        # encoder
        growth_rate: int,
        num_layers: int,
        # decoder
        nhead: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        dc: int,
        cross_coverage: bool,
        self_coverage: bool,
        # beam search
        beam_size: int,
        max_len: int,
        alpha: float,
        early_stopping: bool,
        temperature: float,
        # training
        learning_rate: float,
        patience: int,
        milestones: List[int] = [40, 55],
        dynamic_weight: bool = True,
        vocab_size: int = 114,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.ical_model = ICAL(
            d_model=d_model,
            growth_rate=growth_rate,
            num_layers=num_layers,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            dc=dc,
            cross_coverage=cross_coverage,
            self_coverage=self_coverage,
            vocab_size=vocab_size,
        )

        self.cer_recorder = CERRecorder()
        self.exprate_recorder = ExpRateRecorder()
        self.fps_recorder = FPSRecorder()

    def forward(
        self, img: FloatTensor, img_mask: LongTensor, tgt: LongTensor
    ) -> FloatTensor:
        """run img and bi-tgt

        Parameters
        ----------
        img : FloatTensor
            [b, 1, h, w]
        img_mask: LongTensor
            [b, h, w]
        tgt : LongTensor
            [2b, l]

        Returns
        -------
        FloatTensor
            [2b, l, vocab_size]
        """
        return self.ical_model(img, img_mask, tgt)

    def training_step(self, batch: Batch, _):
        fusion_tgt, fusion_out = plicit_tgt_out(
            batch.indices, self.device, is_explicit=False, is_implicit=False
        )
        exp_tgt, exp_out = plicit_tgt_out(
            batch.indices, self.device, is_explicit=False, is_implicit=False
        )
        implicit_tgt, implicit_out = plicit_tgt_out(
            batch.indices, self.device, is_explicit=False, is_implicit=True
        )

        exp_out_hat, imp_out_hat, fusion_out_hat = self(batch.imgs, batch.mask, exp_tgt)

        exp_loss = ce_loss(exp_out_hat, exp_out)
        implicit_loss = ce_loss(
            imp_out_hat, implicit_out, need_weight=self.hparams.dynamic_weight
        )
        fusion_loss = ce_loss(fusion_out_hat, fusion_out)

        self.log(
            "train_implicit_loss",
            implicit_loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=len(batch.imgs),
        )
        self.log(
            "train_explicit_loss",
            exp_loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=len(batch.imgs),
        )
        self.log(
            "train_fusion_loss",
            fusion_loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=len(batch.imgs),
        )

        loss = exp_loss + implicit_loss + fusion_loss
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=len(batch.imgs),
        )

        return loss

    def validation_step(self, batch: Batch, _):
        fusion_tgt, fusion_out = plicit_tgt_out(
            batch.indices, self.device, is_explicit=False, is_implicit=False
        )
        exp_tgt, exp_out = plicit_tgt_out(
            batch.indices, self.device, is_explicit=False, is_implicit=False
        )
        implicit_tgt, implicit_out = plicit_tgt_out(
            batch.indices, self.device, is_explicit=False, is_implicit=True
        )

        exp_out_hat, imp_out_hat, fusion_out_hat = self(batch.imgs, batch.mask, exp_tgt)

        exp_loss = ce_loss(exp_out_hat, exp_out)
        implicit_loss = ce_loss(
            imp_out_hat, implicit_out, need_weight=self.hparams.dynamic_weight
        )
        fusion_loss = ce_loss(fusion_out_hat, fusion_out)

        self.log(
            "val_fusion_loss",
            fusion_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=len(batch.imgs),
        )
        self.log(
            "val_exp_loss",
            exp_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            batch_size=len(batch.imgs),
        )
        self.log(
            "val_imp_loss",
            implicit_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            batch_size=len(batch.imgs),
        )

        hyps = self.approximate_joint_search(batch.imgs, batch.mask)

        self.exprate_recorder([h.seq for h in hyps], batch.indices)
        self.log(
            "val_ExpRate",
            self.exprate_recorder,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=len(batch.imgs),
        )

    def test_step(self, batch: Batch, _):
        start = perf_counter_ns()

        hyps = self.approximate_joint_search(batch.imgs, batch.mask)
        self.cer_recorder([h.seq for h in hyps], batch.indices)
        self.exprate_recorder([h.seq for h in hyps], batch.indices)

        end = perf_counter_ns()

        self.fps_recorder([end - start for _ in hyps])

    def predict_step(self, batch: Batch, _):
        start = perf_counter_ns()

        hyps = self.approximate_joint_search(batch.imgs, batch.mask)
        self.cer_recorder([h.seq for h in hyps], batch.indices)
        self.exprate_recorder([h.seq for h in hyps], batch.indices)

        results = []

        for hyp, idx, name in zip(hyps, batch.indices, batch.img_bases):
            output = vocab.indices2label(hyp.seq)
            target = vocab.indices2label(idx)

            output = dict(tex=output)
            target = dict(tex=target, name=name)
            sample = dict(outputs=output, targets=target)

            results.append(sample)

        end = perf_counter_ns()

        self.fps_recorder([end - start for _ in hyps])

        return results

    def on_test_epoch_end(self) -> None:
        self.cer = float(self.cer_recorder.compute())
        self.exp_rate = float(self.exprate_recorder.compute())
        self.fps = float(self.fps_recorder.compute())

    def on_predict_epoch_end(self) -> None:
        self.cer = float(self.cer_recorder.compute())
        self.exp_rate = float(self.exprate_recorder.compute())
        self.fps = float(self.fps_recorder.compute())

    def approximate_joint_search(
        self, img: FloatTensor, mask: LongTensor
    ) -> List[Hypothesis]:
        return self.ical_model.beam_search(img, mask, **self.hparams)

    def configure_optimizers(self):
        optimizer = optim.SGD(
            self.parameters(),
            lr=self.hparams.learning_rate,
            momentum=0.9,
            weight_decay=1e-4,
        )

        reduce_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.25,
            patience=self.hparams.patience // self.trainer.check_val_every_n_epoch,
        )
        scheduler = {
            "scheduler": reduce_scheduler,
            "monitor": "val_ExpRate",
            "interval": "epoch",
            "frequency": self.trainer.check_val_every_n_epoch,
            "strict": True,
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def gen_implicit_tokens(self, batch: Batch, hyps: List[Hypothesis]):
        hyps_list = [h.seq for h in hyps]
        fusion_tgt, fusion_out = plicit_tgt_out(
            hyps_list, self.device, is_explicit=False, is_implicit=False
        )
        exp_tgt, exp_out = plicit_tgt_out(
            hyps_list, self.device, is_explicit=False, is_implicit=False
        )
        implicit_tgt, implicit_out = plicit_tgt_out(
            hyps_list, self.device, is_explicit=False, is_implicit=True
        )

        exp_out_hat, imp_out_hat, fusion_out_hat = self(batch.imgs, batch.mask, exp_tgt)
        _, max_indices = torch.max(imp_out_hat, dim=2)
        max_indices = max_indices[:4, :].tolist()
        return max_indices
