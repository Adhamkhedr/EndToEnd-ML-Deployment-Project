from dataclasses import dataclass
import os
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "raw.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")


class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        df = pd.read_csv("notebooks/data/student.csv")

        # Save raw data
        df.to_csv(self.config.raw_data_path, index=False)

        # Split into train/test
        train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

        # Save train and test files
        train_set.to_csv(self.config.train_data_path, index=False)
        test_set.to_csv(self.config.test_data_path, index=False)

        # RETURN THE FILE PATHS:
        return (
            self.config.train_data_path,
            self.config.test_data_path
        )
