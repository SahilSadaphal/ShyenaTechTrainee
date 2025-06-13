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
from sklearn.naive_bayes import MultinomialNB
import joblib

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
    logger.debug("Load_data Function Initiated")
    try:
        train_df = pd.read_csv(path)
        logger.debug("Data Loaded Successfully")
        return train_df
    except Exception as e:
        logger.debug(f"Error occurred {e}")


def save_model(model, path: str):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(model, path)
        logger.debug(f"Model saved successfully at {path}")
    except Exception as e:
        logger.error(f"Error occurred while saving model: {e}")
        raise


def train_model(data: pd.DataFrame):
    logger.debug("train_model function Initiated")
    try:
        model = MultinomialNB()
        X_train = data.iloc[:, :-1].values
        y_train = data.iloc[:, -1].values
        model.fit(X_train, y_train)
        logger.info("Model trained successfully")

        model_save_path = "models/model.joblib"
        save_model(model, model_save_path)
    except Exception as e:
        logger.error(f"Error occurred during model training: {e}")
        raise


def main():
    data_path = r"D:\ShyenaTechTrainee\data\final\final_vector_train.csv"
    train_df = load_data(data_path)
    train_model(train_df)


if __name__ == "__main__":
    main()
