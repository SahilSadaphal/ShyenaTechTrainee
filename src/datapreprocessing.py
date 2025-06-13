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
from nltk.stem.porter import PorterStemmer
from sklearn.preprocessing import LabelEncoder

nltk.download("punkt")
nltk.download("stopwords")


log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("DataPreprocessing")
logger.setLevel("DEBUG")

# Clear existing handlers if rerunning
if logger.hasHandlers():
    logger.handlers.clear()

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

file_path = os.path.join(log_dir, "DataPreprocessing.log")
file_handler = logging.FileHandler(file_path, mode="a")  # mode='a' for append
file_handler.setLevel("DEBUG")

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def preprocess(text: str):
    try:
        ps = PorterStemmer()
        text = text.lower()
        text = nltk.word_tokenize(text)
        text = [word for word in text if word.isalnum()]
        text = [
            word
            for word in text
            if word not in stopwords.words("english") and word not in string.punctuation
        ]
        text = [ps.stem(word) for word in text]
        return " ".join(text)
    except Exception as e:
        logger.error(f"Error occurred during preprocessing: {e}")
        return ""


def encode_data(
    data: pd.DataFrame, text_column: str, target_column: str
) -> pd.DataFrame:
    try:
        logger.debug("Starting Preprocessing for DF")
        le = LabelEncoder()
        data[target_column] = le.fit_transform(data[target_column])

        # Removing Duplicates
        data = data.drop_duplicates(keep="first")
        logger.debug("Dropped Duplicate rows from df")
        logger.debug("Preprocess Function Started for text column")
        data[text_column] = data[text_column].apply(preprocess)

        logger.debug("Text column transformed")
        return data
    except Exception as e:
        logger.error(f"Error Occured {e}")


def main():
    train = pd.read_csv(r"D:\ShyenaTechTrainee\data\raw\train.csv")
    test = pd.read_csv(r"D:\ShyenaTechTrainee\data\raw\test.csv")
    logger.debug("train preprocessing started")
    pro_train = encode_data(train, "text", "target")
    logger.debug("Test preprocessing started")
    pro_test = encode_data(test, "text", "target")
    logger.debug("Train & Test Data Transformed Successfully")
    datapath = os.path.join("./data", "interim")
    os.makedirs(datapath, exist_ok=True)
    pro_train.to_csv(os.path.join(datapath, "pro_train.csv"), index=False)
    pro_test.to_csv(os.path.join(datapath, "pro_test.csv"), index=False)


if __name__ == "__main__":
    main()
