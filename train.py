from pytorch_lightning.cli import LightningCLI

from ical.datamodule import CROHMEDatamodule
from ical.lit_ical import LitICAL

cli = LightningCLI(
    LitICAL,
    CROHMEDatamodule,
)
