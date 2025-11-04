import pickle
from pathlib import Path

from gryph.inkml import paint_inkml, scale_inkml
from torch.utils.data.dataset import Dataset


class CROHMEDataset(Dataset):
    def __init__(
        self,
        path: str,
        split: str,
        w: int,
        h: int,
        fill: int,
        line: int,
    ):
        super().__init__()

        assert isinstance(split, str)

        with Path(path).expanduser().open("rb") as f:
            self.ds = pickle.load(f).get(split, [])

        assert isinstance(w, int)
        assert isinstance(h, int)

        assert isinstance(fill, int)
        assert isinstance(line, int)

        self.w = w
        self.h = h

        self.fill = fill
        self.line = line

    def __getitem__(self, idx):
        item = self.ds[idx]

        tex = item["tex"]
        ink = item["ink"]

        ink = scale_inkml(ink, w=self.w, h=self.h)
        img = paint_inkml(ink, w=self.w, h=self.h, fill=self.fill, line=self.line)

        return item["name"], img, tex

    def __len__(self):
        return len(self.ds)
