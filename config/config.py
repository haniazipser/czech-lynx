from pathlib import Path
from typing import Literal

from pydantic import BaseModel
#common config
class DataConfig(BaseModel):
    data_root: Path = Path("data/kaggle-data")
    csv_path: Path = Path("data/kaggle-data/CzechLynxDataset-Metadata-Real.csv")
    split_type: Literal["geo_aware", "time_open", "time_closed"] = "geo_aware"
    image_size: int = 224
    batch_size: int = 64
    num_workers: int = 4
    experiment_type: Literal["baseline", "metric_learning"]
    epochs: int = 15
    lr: float = 1e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 0
    warmup_start: float  = 0.0
    warmup_end: float = 0.0