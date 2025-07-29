@dataclass
class DataTransformationConfig:
preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()  # Load path to save preprocessor.pkl
        self.logger = get_logger(__name__)        # Set up logging

def get_preprocessor(self, numeric_columns, categorical_columns):   #function gets called in initiate_data_transformation
    """
    Build a ColumnTransformer with pipelines for numeric and categorical features.
    """
    # Pipeline for numeric features
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    # Pipeline for categorical features
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ("scaler", StandardScaler(with_mean=False))
    ])
    # Combine both pipelines into one transformer
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, numeric_columns),    #numeric_columns will be defined in initiate_data_transformation function
        ("cat", categorical_pipeline, categorical_columns)   #categorical_columns will be defined in initiate_data_transformation function
    ])



def initiate_data_transformation(self, train_path, test_path):
    """
    Transforms training and testing data using preprocessing pipeline.
    Returns transformed arrays and path to saved preprocessor object.
    """

    # Load datasets
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Set your actual target column here
    target_column = "target"

    # Separate input features and target
    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

 
   numeric_columns = []  
   categorical_columns = []

    for i in X_train.columns:
        if X_train[i].dtype in ["int64", "float64"]:
            numeric_columns.append(i)
        elif X_train[i].dtype == "object":
            categorical_columns.append(i)

    # Build preprocessor
    preprocessor = self.get_preprocessor(numeric_columns, categorical_columns)

    # Apply transformation
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # Save the fitted preprocessor object
    with open(self.config.preprocessor_obj_file_path, "wb") as f:
        pickle.dump(preprocessor, f)

    # Combine transformed features with targets
    train_arr = np.c_[X_train_transformed, y_train]
    test_arr = np.c_[X_test_transformed, y_test]

    return train_arr, test_arr, self.config.preprocessor_obj_file_path
