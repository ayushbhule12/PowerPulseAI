# src/load_data.py

from pathlib import Path
import io
import zipfile
import urllib.request

import pandas as pd


UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/"
    "household_power_consumption.zip"
)


def download_dataset(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset from UCI to {destination}...")
    response = urllib.request.urlopen(UCI_URL)
    with zipfile.ZipFile(io.BytesIO(response.read())) as z:
        inner_name = "household_power_consumption.txt"
        with z.open(inner_name) as zipped_file, destination.open("wb") as out_file:
            out_file.write(zipped_file.read())

    return destination


def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Load the UCI household power consumption dataset.

    The file uses semicolons as separators and '?' for missing values.
    This function parses the Date and Time columns into a datetime index
    without using deprecated pandas arguments.
    """
    path = Path(filepath)
    if not path.exists():
        path = download_dataset(path)

    df = pd.read_csv(
        path,
        sep=';',
        na_values='?',
        low_memory=False,
        dtype=str,
    )

    df['datetime'] = pd.to_datetime(
        df['Date'] + ' ' + df['Time'],
        dayfirst=True,
        format='%d/%m/%Y %H:%M:%S',
        errors='coerce',
    )
    df.set_index('datetime', inplace=True)

    df.drop(columns=['Date', 'Time'], inplace=True)
    df = df.apply(pd.to_numeric, errors='coerce')

    return df


if __name__ == "__main__":
    df = load_raw_data("data/household_power_consumption.txt")
    print(df.shape)
    print(df.dtypes)
    print(df.head())