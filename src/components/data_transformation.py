import os
import pandas as pd
import numpy as np
import pickle
import sys
from dataclasses import dataclass
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.utils.logging import get_logger
from src.utils.exception_handling import CustomException

#same idea as ingestion.py : DataTransformationConfig to store the file path preprocessor_obj_file_path, 
#where Python object preprocessor (which knows how to clean and transform any data) will be saved as a .pkl file.

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:  #contains all the logic for transforming the data.
    def __init__(self):      #Setup Config and Logger
        self.config = DataTransformationConfig()  # have access to the preprocessor_obj_file_path
        self.logger = get_logger(__name__)        # Set up logging

    def get_preprocessor(self, numeric_columns, categorical_columns):   #function gets called in initiate_data_transformation
        """
        Builds a ColumnTransformer with pipelines for numeric and categorical features.
        """
        # Pipeline for numeric features
        numeric_pipeline = Pipeline(steps=[
            ("fill_missing", SimpleImputer(strategy="median")),
            ("scale_numeric", StandardScaler())
        ])
        # Pipeline for categorical features
        categorical_pipeline = Pipeline(steps=[
            ("fill_missing", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ("scale_cat", StandardScaler(with_mean=False))  #dont subtract the mean since they are 
            #encoded so a 0 value may crash everything 
        ])
        # Combine both pipelines into one transformer
        preprocessor = ColumnTransformer(transformers=[
            ("num", numeric_pipeline, numeric_columns),    #numeric_columns will be defined in initiate_data_transformation function
            ("cat", categorical_pipeline, categorical_columns)   #categorical_columns will be defined in initiate_data_transformation function
        ])

        return preprocessor


#initiate_data_transformation should Transform training and testing data using preprocessing pipeline.
# and Return transformed arrays and path to saved preprocessor object.
    def initiate_data_transformation(self, train_path, test_path):
        try:
            self.logger.info("Starting data transformation...")

            # Load datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            self.logger.info("Loaded train and test datasets.")

            # Add total and average score and use average as target instead of the 3 scores 
            for df in [train_df, test_df]:
                df["total_score"] = df["math_score"] + df["reading_score"] + df["writing_score"]
                df["average_score"] = df["total_score"] / 3

            # Add score_diff feature to test consistency amon student grades
            for df in [train_df, test_df]:
                df["score_diff"] = df[["math_score", "reading_score", "writing_score"]].max(axis=1) - \
                                df[["math_score", "reading_score", "writing_score"]].min(axis=1)

            # Define label functions corrsoponding to target average score and difference in score feature 
            def get_perf_label(avg):   #shows Real gaps in performance
                if avg >= 85:
                    return "High"
                elif avg >= 70:
                    return "Medium"
                else:
                    return "Low"

            def get_consistency_label(diff): 
                if diff <= 5:
                    return "Very Consistent"
                elif diff <= 15:
                    return "Moderately Consistent"
                else:
                    return "Inconsistent"

            # Apply labels
            for df in [train_df, test_df]:
                df["performance_label"] = df["average_score"].apply(get_perf_label)
                df["consistency_label"] = df["score_diff"].apply(get_consistency_label)
                df["summary_label"] = df["performance_label"] + " — " + df["consistency_label"]

            # Now set target
            target_column = "average_score"

            # Separate input features and target
            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]
            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]

            # Determine numeric and categorical columns
            numeric_columns = []
            categorical_columns = []
        #add numeric features to numeric_columns and categorical features to categorical_columns
            for col in X_train.columns:
                if X_train[col].dtype in ["int64", "float64"]:
                    numeric_columns.append(col)
                elif X_train[col].dtype == "object":
                    categorical_columns.append(col)

            if "summary_label" in categorical_columns:
                categorical_columns.remove("summary_label")

            # Build preprocessor
            self.logger.info("Building preprocessing pipeline...")
            preprocessor = self.get_preprocessor(numeric_columns, categorical_columns)

            # Apply transformation
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            # Save the fitted preprocessor object
            with open(self.config.preprocessor_obj_file_path, "wb") as f:
                pickle.dump(preprocessor, f)
            self.logger.info(f"Saved preprocessor object to: {self.config.preprocessor_obj_file_path}")

            # Combine transformed features with targets to stack arrays side by side
            train_arr = np.c_[X_train_transformed, y_train]
            test_arr = np.c_[X_test_transformed, y_test]
            self.logger.info("Data transformation completed successfully.")

            return train_arr, test_arr, self.config.preprocessor_obj_file_path

        except Exception as e:
            self.logger.error("Exception occurred during data transformation.")
            raise CustomException(e, sys)
