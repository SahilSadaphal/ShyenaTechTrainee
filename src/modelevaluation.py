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
import os
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import logging
import yaml
from joblib import load

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logger = logging.getLogger("ModelEvaluation")
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")
console_handler.setFormatter(formatter)

log_file_path = os.path.join(log_dir, "ModelEvaluation.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_data(path):
    try:
        logger.debug("load data function initiated")
        test_df = pd.read_csv(path)
        return test_df
    except Exception as e:
        logger.error(f"Error Occured {e}")


def load_model(path):
    try:
        logger.debug("load_model func initiated")
        with open(path, "rb") as file:
            model = load(file)
            logger.debug("Model Loaded Successfully")
            return model
    except Exception as e:
        logger.error(f"Error Occurred {e}")


def evaluate(data, model):
    try:
        logger.debug("evaluate function initiated")
        X_test = data.iloc[:, :-1].values
        y_test = data.iloc[:, -1].values
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        metric = {"accuracy": accuracy, "precision": precision, "recall": recall}
        return metric
    except Exception as e:
        logger.error(f"Error Occured {e}")
        raise


def main():
    data_path = r"D:\ShyenaTechTrainee\data\final\final_vector_test.csv"
    model_path = r"D:\ShyenaTechTrainee\models\model.joblib"
    try:
        test_df = load_data(data_path)
        model = load_model(model_path)
        metric = evaluate(test_df, model)
        os.makedirs("metric", exist_ok=True)
        file_path = os.path.join("metric", "metric.json")
        with open(file_path, "w") as file:
            json.dump(metric, file, indent=4)

        logger.debug("Metrics saved to %s", file_path)
    except Exception as e:
        logger.error(f"Error Occured {e}")


if __name__ == "__main__":
    main()
