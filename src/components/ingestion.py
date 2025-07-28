from dataclasses import dataclass    #shortcut
import os
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from src.utils.logging import get_logger
from src.utils.exception_handling import CustomException


logger = get_logger(__name__)

@dataclass
class DataIngestionConfig:     #Instead of hardcoring the file paths everytime , config holds the 3 file paths 
    raw_data_path: str = os.path.join("artifacts", "raw.csv")    #os.path.join() to join file paths
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")

"""
    Handles data ingestion:
    - Loads raw CSV data
    - Saves untouched copy
    - Splits into train/test
    - Saves both sets
    - Returns file paths
    """
class DataIngestion:    

    def __init__(self):    #automatically runs when creating the object from a class.
        self.config = DataIngestionConfig()     #Set up the config  
        #now I can access:
        #self.config.raw_data_path  → "artifacts/raw.csv"
        #self.config.train_data_path → "artifacts/train.csv"


    def initiate_data_ingestion(self):   #actual ingestion process 
        logger.info("Starting data ingestion process")
        try:
            df = pd.read_csv("notebooks/data/student.csv")    #load the raw - not adjusted data 
            logger.info("Dataset successfully loaded")
            os.makedirs(os.path.dirname(self.config.raw_data_path), exist_ok=True)
            # Save raw data
            df.to_csv(self.config.raw_data_path, index=False)     #keeping a backup of the untouched data.
            logger.info(f"Raw data saved to: {self.config.raw_data_path}")
            # Split into train/test
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)  
            logger.info("Train/test split completed")
            # Save train and test files
            train_set.to_csv(self.config.train_data_path, index=False)     
            test_set.to_csv(self.config.test_data_path, index=False)
                                                                        #This saves them to:
                                                                        #artifacts/train.csv
                                                                        #artifacts/test.csv
            logger.info(f"Train data saved to: {self.config.train_data_path}")
            logger.info(f"Test data saved to: {self.config.test_data_path}")

        # RETURN THE FILE PATHS:
            return (
                self.config.train_data_path,
                self.config.test_data_path
             )

        except Exception as e:
            logger.error("Error occurred during data ingestion", exc_info=True)
            raise CustomException(str(e), sys)