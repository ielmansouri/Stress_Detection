import os
import pandas as pd
import numpy as np
import rampwf as rw

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import precision_score, recall_score,accuracy_score,f1_score,classification_report

#-----------------------------------------------------------------------
problem_title = 'stress rate prediction'
_target_column_name = 'Label'
_prediction_label_names = [0, 1, 2, 3, 4]
# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_multiclass(
    label_names=_prediction_label_names)

# An object implementing the workflow
workflow = rw.workflows.FeatureExtractorClassifier()

#-----------------------------------------------------------------------
# Define custom score metrics for the churner class
class Precision(rw.score_types.classifier_base.ClassifierBaseScoreType):

    def __init__(self, name='precision', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        return accuracy_score(y_true, y_pred)
    
class Recall(rw.score_types.classifier_base.ClassifierBaseScoreType):

    def __init__(self, name='recall', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        return recall_score(y_true, y_pred,average= 'macro')
    
class f1(rw.score_types.classifier_base.ClassifierBaseScoreType):

    def __init__(self, name='f1_score', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        return f1_score(y_true, y_pred,average= 'macro')
class prec2(rw.score_types.classifier_base.ClassifierBaseScoreType):

    def __init__(self, name='Precision_Stress', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        return classification_report(y_true, y_pred,output_dict=True)['2']['recall']
class final_score(rw.score_types.classifier_base.ClassifierBaseScoreType):

    def __init__(self, name='final_score', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        return 0.6*classification_report(y_true, y_pred,output_dict=True)['2']['recall']+0.4*f1_score(y_true, y_pred,average= 'macro')

score_types = [
    Recall(),
    Precision(),
    f1(),
    prec2(),
    final_score(),
]

#-----------------------------------------------------------------------
def get_cv(X, y):
    """Returns stratified randomized folds."""
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.25, random_state=57)
    return cv.split(X,y)

def _read_data(path, f_name):
    data = pd.read_csv(os.path.join(path, 'data', f_name), sep=',')
    y_array = data[_target_column_name].values
    X_df = data.drop(['Unnamed: 0','Label'],axis=1)
    test = os.getenv('RAMP_TEST_MODE', 0)
    if test:
        return X_df[:200], y_array[:200]
    else:
        return X_df, y_array


def get_train_data(path='.'):
    f_name = 'train.csv'
    return _read_data(path, f_name)


def get_test_data(path='.'):
    f_name = 'test.csv'
    return _read_data(path, f_name)
