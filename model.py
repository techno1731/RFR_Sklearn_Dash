import pandas as pd
import pickle

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Defining functions for use in training, testing and deployment.

# preprocess the Origin column in data
def preprocess_origin_cols(df):
    """
    Maps numerical categories in Origin column to its country
    of origin full name.

    Argument:
        df: Original DataFrame
    Returns:
        df: Mapped DataFrame.
    """
    df["Origin"] = df["Origin"].map({1: "India", 2: "USA", 3: "Germany"})
    return df


# preprocess the numerical data
def num_pipeline_transformer(data):
    """
    Function to process numerical transformations
    Argument:
        data: original DataFrame
    Returns:
        num_attrs: numerical dataframe
        num_pipeline: numerical pipeline object

    """
    numerics = ["float64", "int64"]

    num_attrs = data.select_dtypes(include=numerics)

    num_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("std_scaler", StandardScaler()),
        ]
    )
    return num_attrs, num_pipeline


# preprocess the cat data and concatenate with num data
def pipeline_transformer(data):
    """
    Transformation pipeline requires num_pipeline_transformer
    preprocess_origin_cols.

    Argument:
        data: output of preprocess_origin_cols
    Returns:
        prepared_data: numpy array, ready to use by model.bin
    """
    cat_attrs = ["Origin"]
    num_attrs, num_pipeline = num_pipeline_transformer(data)
    full_pipeline = ColumnTransformer(
        [
            ("num", num_pipeline, list(num_attrs)),
            ("cat", OneHotEncoder(), cat_attrs),
        ]
    )
    prepared_data = full_pipeline.fit_transform(data)
    return prepared_data


# deal with Origin column for one step preprocessing
def full_preprocess(dataframe):
    """
    Full preprocessing pipeline condensed into one function
    requires preprocess_origin_cols, num_pipeline_transformer
    and pipeline_transformer.

    Argument:
        dataframe: Original Dataframe.
    Returns:
        step_two: numpy array, scaled and onehot encoded
        ready for use with model.bin
    """
    step_one = preprocess_origin_cols(dataframe)
    step_two = pipeline_transformer(step_one)
    return step_two


def load_model():
    """
    Load the model.bin file for use in the application

    Argument:
        None
    Returns:
        model: the model object
    Usage:
        model = load_model()
    """
    ##loading the model from the saved file
    with open("model.bin", "rb") as f_in:
        model = pickle.load(f_in)
    return model


# predict function for ease of use with json objects
def predict_mpg(config, model):
    """
    Function used to predict new data points from a Dict like object

    Argument:
        config: Dict like object contining all the keys from original
        DataFrame.
        model: The model loaded using load_model function.
    Returns:
        y_pred: List of predictions

    Known Issues:
        Due to the ColumnTransformer method, the input dict should contain
        at least 3 values per key, otherwise the size of the numpy
        array is reduced by one or two elements and the model.bin
        will return and error fo dimensionality expecting 9 elements
        and recieving 7 or 8.
    """
    df = pd.DataFrame(config)

    prepared_df = full_preprocess(df)
    y_pred = model.predict(prepared_df)
    return y_pred
