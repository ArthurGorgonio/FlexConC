import pandas as pd
# from skmultiflow.data import FileStream


def read_stream(filename, path='../datasets/'):
    stream = FileStream(path+filename)
    
    return stream


def read_data(filename, path='../datasets/'):
    df = pd.read_csv(path+filename)
    
    return df


def split_data(df: pd.DataFrame, ratio=0.1):
    """[summary]

    Args:
        df (pd.DataFrame): The dataset with 
        ratio (float, optional): A number between [0,1] to determine the ratio of labelled data. Defaults to 0.1.
    """
    pass

def validate_estimator(estimator):
    """Make sure that an estimator implements the necessary methods."""
    if not hasattr(estimator, "predict_proba"):
        msg = "base_estimator ({}) should implement predict_proba!"
        raise ValueError(msg.format(type(estimator).__name__))