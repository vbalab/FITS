import random
from enum import Enum
from pathlib import Path

import numpy as np
import torch

_DIR_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_PATH = _DIR_ROOT / "data/"
DATASETS_PATH = DATA_PATH / "datasets/"
MODELS_PATH = DATA_PATH / "models/"
TRAINING_PATH = MODELS_PATH / "training/"
EVALUATION_PATH = MODELS_PATH / "evaluation/"

DATA_PATH.mkdir(parents=True, exist_ok=True)
DATASETS_PATH.mkdir(parents=True, exist_ok=True)
MODELS_PATH.mkdir(parents=True, exist_ok=True)
TRAINING_PATH.mkdir(parents=True, exist_ok=True)
EVALUATION_PATH.mkdir(parents=True, exist_ok=True)


class DatasetsPaths(Enum):
    solar = DATASETS_PATH / "solar/solar_AL.txt"
    etth = DATASETS_PATH / "etth/ETTh1.csv"
    electricity = DATASETS_PATH / "electricity/electricity.txt"
    exchange = DATASETS_PATH / "exchange/exchange_rate.txt"
    weather = DATASETS_PATH / "weather/weather.csv"


def SeedEverything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    print(f"Global seed set to {seed}")
