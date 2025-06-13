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

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logger = logging.getLogger("FeatureEngineering")
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")
console_handler.setFormatter(formatter)

log_file_path = os.path.join(log_dir, "FeatureEngineering.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_data(path: str) -> pd.DataFrame:
    try:
        logger.debug("load_data Function Initiated")
        df = pd.read_csv(path)
        df.fillna("", inplace=True)
        return df
    except Exception as e:
        logger.error(f"Error Occured {e}")
        raise


def apply_tfidf(train: pd.DataFrame, test: pd.DataFrame, max_features) -> pd.DataFrame:
    try:
        logger.debug("apply_tfidf function started")
        X_train = train["text"].values
        X_test = test["text"].values
        y_train = train["target"].values
        y_test = test["target"].values
        tfidf = TfidfVectorizer(max_features=max_features)
        X_train_bow = tfidf.fit_transform(X_train)
        X_test_bow = tfidf.transform(X_test)

        train_df = pd.DataFrame(X_train_bow.toarray())
        test_df = pd.DataFrame(X_test_bow.toarray())
        train_df["label"] = y_train
        test_df["label"] = y_test
        logger.debug("Applied_tfidf Successfully")
        return train_df, test_df
    except Exception as e:
        logger.error(f"Error Occured {e}")
        raise


def main():
    max_features = 29
    try:
        train_df = load_data(r"D:\ShyenaTechTrainee\data\interim\pro_train.csv")
        test_df = load_data(r"D:\ShyenaTechTrainee\data\interim\pro_test.csv")
        logger.info("Data Loaded Succcessfully")
        vector_train, vector_test = apply_tfidf(train_df, test_df, 29)
        data_file_path = os.path.join("./data", "final")
        os.makedirs(data_file_path, exist_ok=True)
        vector_train.to_csv(
            os.path.join(data_file_path, "final_vector_train.csv"), index=False
        )
        vector_test.to_csv(
            os.path.join(data_file_path, "final_vector_test.csv"), index=False
        )

    except Exception as e:
        logger.error(f"Error Occured {e}")
        raise


if __name__ == "__main__":
    main()
