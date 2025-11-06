import argparse
from pathlib import Path
from pickle import dumps
from typing import Sequence

from gryph.score import CER
from pytorch_lightning import Trainer, seed_everything

from ical.datamodule import CROHMEDatamodule
from ical.lit_ical import LitICAL

seed_everything(7)


def options():
    args = argparse.ArgumentParser()

    args.add_argument("--ckpt", type=str, required=True)
    args.add_argument("--data", type=str, required=True)
    args.add_argument("--store", type=str, required=True)
    args.add_argument("--splits", type=str, nargs="+", required=True)

    return vars(args.parse_args())


def process(ckpt: str, data: str, split: str):
    result = dict(ckpt=ckpt, data=data, split=split)

    trainer = Trainer(logger=False, devices=1)

    dm = CROHMEDatamodule(
        path=data,
        test_split=split,
        w=224,
        h=224,
        fill=255,
        line=1,
        train_batch_size=4,
        eval_batch_size=4,
        num_workers=1,
    )

    model = LitICAL.load_from_checkpoint(ckpt)
    data = sum(trainer.predict(model, datamodule=dm), [])

    params = sum(p.numel() for p in model.parameters())

    scores = CER(prefix="tex").compute_metrics(data)
    scores.update(fps=model.fps, params=params)

    result.update(scores=scores, data=data)

    return result


def main(ckpt: str, data: str, store: str, splits: Sequence[str]):
    result = {split: process(ckpt, data, split) for split in splits}

    path = Path(store).expanduser()
    path.write_bytes(dumps(result))


if __name__ == "__main__":
    main(**options())
