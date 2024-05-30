import mlflow
import pickle
from sklearn.metrics import root_mean_squared_error
from typing import Tuple
from sklearn.base import BaseEstimator
from pandas import Series
from scipy.sparse._csr import csr_matrix
from sklearn.feature_extraction import DictVectorizer
from mlflow.tracking import MlflowClient


MLFLOW_TRACKING_URI = "sqlite:///mlflow.db" #'sqlite:///home/mlflow/mlflow.db'

DEFAULT_EXPERIMENT_NAME = 'nyc-taxi-experiment-mage'

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def export_data(
    trained_model: Tuple[DictVectorizer, BaseEstimator], **kwargs):
    #, csr_matrix, Series
    # read in model and dv from previous step
    dv, model = trained_model
    #dv, model, X_train, y_train= trained_model

    # setup experiment
    mlflow.set_tracking_uri("http://mlflow:5000")
    #mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(DEFAULT_EXPERIMENT_NAME)
    #client = MlflowClient()
    #experiment = client.get_experiment_by_name(DEFAULT_EXPERIMENT_NAME)
    #experiment_id = experiment.experiment_id

    # start a run
    with mlflow.start_run():
        mlflow.set_tag("developer","andrea")

        # log info about dataset
        mlflow.log_param("train-data-path","yellow_tripdata_2023-03.parquet")

        #calculate metric and log
        #y_pred = model.predict(X_train)
        #rmse = root_mean_squared_error(y_train,y_pred)
        #mlflow.log_metric("rmse",rmse)

        # save preprocessor as pickle file
        #with open("../preprocessor.b","wb") as f_out:
        #    pickle.dump(dv, f_out)

        # log the preprocessors as artifacts
        # mlflow.log_artifact("../preprocessor.b",artifact_path="preprocessor")

        # log model
        mlflow.sklearn.log_model(model,artifact_path='models')
    
    print('done.')

   

