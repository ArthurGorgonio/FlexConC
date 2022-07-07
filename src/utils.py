from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

def validate_estimator(estimator):
    """Make sure that an estimator implements the necessary methods."""
    if not hasattr(estimator, "predict_proba"):
        msg = "base_estimator ({}) should implement predict_proba!"
        raise ValueError(msg.format(type(estimator).__name__))

# Spliting train ans test data
def crossValidation(k):
    try:
        return StratifiedKFold(n_splits=k)
    except ValueError:
        print("Please put a number > 2")