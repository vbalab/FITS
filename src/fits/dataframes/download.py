import gzip
import shutil

import requests

from fits.config import DATASETS_PATH, DatasetsPaths


def DownloadDatasetSolar() -> None:
    url = "https://github.com/laiguokun/multivariate-time-series-data/raw/master/solar-energy/solar_AL.txt.gz"

    archive_path = DATASETS_PATH / "solar_AL.txt.gz"
    dataset_path = DatasetsPaths.solar.value
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(archive_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    with gzip.open(archive_path, "rb") as gz_file:
        with open(dataset_path, "wb") as out_file:
            shutil.copyfileobj(gz_file, out_file)

    archive_path.unlink()


def DownloadDatasetETTh() -> None:
    url = (
        "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
    )

    dataset_path = DatasetsPaths.etth.value
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dataset_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)


def DownloadDatasetElectricity() -> None:
    url = "https://github.com/laiguokun/multivariate-time-series-data/raw/master/electricity/electricity.txt.gz"

    archive_path = DATASETS_PATH / "electricity.txt.gz"
    dataset_path = DatasetsPaths.electricity.value
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(archive_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    with gzip.open(archive_path, "rb") as gz_file:
        with open(dataset_path, "wb") as out_file:
            shutil.copyfileobj(gz_file, out_file)

    archive_path.unlink()


def DownloadDatasetExchange() -> None:
    url = "https://github.com/laiguokun/multivariate-time-series-data/raw/master/exchange_rate/exchange_rate.txt.gz"

    archive_path = DATASETS_PATH / "exchange_rate.txt.gz"
    dataset_path = DatasetsPaths.exchange.value
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(archive_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    with gzip.open(archive_path, "rb") as gz_file:
        with open(dataset_path, "wb") as out_file:
            shutil.copyfileobj(gz_file, out_file)

    archive_path.unlink()


def DownloadDatasetWeather() -> None:
    url = "https://huggingface.co/datasets/pkr7098/time-series-forecasting-datasets/resolve/main/weather.csv"

    dataset_path = DatasetsPaths.weather.value
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dataset_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
