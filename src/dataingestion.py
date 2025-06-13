import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from sklearn.model_selection import train_test_split
import logging
import yaml

# Create log directory
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Configure logger
logger = logging.getLogger("DataIngestion")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir, "DataIngestion.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def data_ingest(path):
    """
    This Function Fetches the data and saves it to data directory in raw subfolder
    """
    try:
        logger.info("data_ingest function started")
        df = pd.read_csv(path)
        logger.debug(f"Data Loaded Successfully from path: {path}")
        return df
    except Exception as e:
        logger.error(f"Error Occurred: {e}")
        raise


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    This Function Removes Duplicate Rows, Unnecessary Columns and Renames columns
    """
    try:
        logger.info("preprocess_data started")
        df.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], inplace=True)
        df.drop_duplicates(inplace=True)
        df.rename(columns={"v1": "target", "v2": "text"}, inplace=True)
        logger.debug("Data preprocessing completed successfully")
        return df
    except Exception as e:
        logger.error(f"Error Occurred: {e}")
        raise


def save_data(train: pd.DataFrame, test: pd.DataFrame):
    """
    This Function saves train and test data in data/raw directory
    """
    try:
        logger.info("SaveData Function Started")
        path = "data/raw"
        os.makedirs(path, exist_ok=True)
        train.to_csv(os.path.join(path, "train.csv"), index=False)
        test.to_csv(os.path.join(path, "test.csv"), index=False)
        logger.debug(f"Train and Test data successfully saved to :{path}")
    except Exception as e:
        logger.error(f"Error Occurered {e}")


def main():
    df = data_ingest(r"D:\ShyenaTechTrainee\Experiments\spam.csv")
    df = preprocess_data(df)
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    save_data(train, test)


if __name__ == "__main__":
    main()
