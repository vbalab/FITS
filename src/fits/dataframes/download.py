import shutil
import tarfile
import zipfile
import gzip
import requests

from fits.config import DATASETS_PATH, DatasetsPaths


headers = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}


def DownloadDatasetPhysio() -> None:
    url = "https://physionet.org/files/challenge-2012/1.0.0/set-a.tar.gz?download"

    archive_path = DATASETS_PATH / "set-a.tar.gz"

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(archive_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    with tarfile.open(archive_path, mode="r:gz") as t:
        t.extractall(path=DatasetsPaths.physio.value)

    archive_path.unlink()


def DownloadDatasetAirQuality() -> None:
    url = "https://www.microsoft.com/en-us/research/wp-content/uploads/2016/06/STMVL-Release.zip"

    tmp_file = DATASETS_PATH / "STMVL-Release.zip"
    tmp_dir = DATASETS_PATH / "pm25"

    url_data = requests.get(
        url,
        headers=headers,
        stream=True,
        allow_redirects=True,
    ).content

    with open(tmp_file, mode="wb") as f:
        f.write(url_data)

    with zipfile.ZipFile(tmp_file) as z:
        z.extractall(tmp_dir)

    tmp_file.unlink()

    shutil.move(
        tmp_dir / "Code/STMVL/SampleData/pm25_ground.txt",
        DATASETS_PATH / "pm25_ground.txt",
    )
    shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    shutil.move(DATASETS_PATH / "pm25_ground.txt", DatasetsPaths.pm25.value)


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
    ...
