import pickle
from pathlib import Path

from gryph.inkml import paint_inkml, scale_inkml
from mmcv.image import gray2rgb, impad, imrescale
from torch.utils.data.dataset import Dataset
from torchvision.transforms.functional import to_tensor


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

        tex = item.get("tex")
        ink = item.get("ink")
        img = item.get("img")

        if img is not None:
            img = self.process_img(img)

        else:
            img = self.process_ink(ink)

        return item["name"], img, tex

    def process_img(self, img):
        img = gray2rgb(img)
        img = imrescale(img, scale=(self.h, self.w))
        img = impad(img, shape=(self.h, self.w))
        img = to_tensor(img)

        return img

    def process_ink(self, ink):
        ink = scale_inkml(ink, w=self.w, h=self.h)
        img = paint_inkml(ink, w=self.w, h=self.h, fill=self.fill, line=self.line)

        return img

    def __len__(self):
        return len(self.ds)
