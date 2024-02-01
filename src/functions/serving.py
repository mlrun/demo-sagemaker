import tarfile
import warnings
from typing import List

import mlrun
import numpy as np
import pandas as pd
import xgboost as xgb
from cloudpickle import load

warnings.filterwarnings("ignore")


class XGBModelServer(mlrun.serving.V2ModelServer):
    def load(self):
        """load and initialize the model and/or other elements"""
        if not self.model_path:
            self._set_model_path()
        # Download the model file:
        model_file, extra_data = self.get_model(".tar.gz")

        # Extract model file:
        t = tarfile.open(model_file, "r:gz")
        t.extractall()

        # Load model from file:
        self.model = load(open("xgboost-model", "rb"))

    def predict(self, body: dict) -> List:
        """Generate model predictions from sample."""

        # Convert input to numpy array:
        data = np.asarray(body["inputs"])

        # Transform into XGBoost object:
        data = xgb.DMatrix(data)

        # Perform prediction:
        result: np.ndarray = self.model.predict(data)
        return result.tolist()

    def _set_model_path(self):
        # Get project:
        context = mlrun.get_or_create_ctx("mlrun")
        project_name = context.project
        project = mlrun.get_or_create_project(project_name)

        # get model path from artifact:
        model_path_artifact = project.get_artifact("train_model_path")
        model_path = model_path_artifact.to_dataitem().get().decode("utf-8")

        # set model path:
        self.model_path = model_path
        
# Function that preprocesses the inference data
def preprocess(data: pd.Dataframe):
    unique_categories = data.transaction_category.unique()
    # Create a feature vector that gets the average amount
    vector = fstore.FeatureVector("transactions_vector", ["aggregations.amount_avg_1d"], with_indexes=True)

    # Use online feature service to get the latest average amount per category
    with vector.get_online_feature_service() as online_feature_service:
        resp = online_feature_service.get(
            [{"transaction_category":cat} for cat in unique_categories]
        )
    
    for cat in resp:
        transaction_category = cat['transaction_category']
        amount_avg = cat['amount_avg_1d']
        data["dist_" + transaction_category] = abs(amount_avg - data["amount"])
    
    # convert timestamp to components
    data["year"] = data["timestamp"].dt.year
    data["month"] = data["timestamp"].dt.month
    data["day"] = data["timestamp"].dt.day
    data["hour"] = data["timestamp"].dt.hour
    data["minute"] = data["timestamp"].dt.minute
    data["second"] = data["timestamp"].dt.second

    del data["timestamp"]
    del data["transaction_category"]
    
    return data
    
    



def postprocess(inputs: dict) -> dict:
    """
    Postprocessing the output of the model
    """
    # Read the prediction:
    print(inputs)
    outputs = np.asarray(inputs.pop("outputs"))
    predictions = []
    confidences = []

    # Turn predictions into categories and extract confidence:
    for prediction in outputs:
        pred = prediction.argmax()
        confidence = max(prediction)
        predictions.extend([int(pred)])
        confidences.extend([float(confidence)])

    inputs["predictions"] = predictions
    inputs["confidences"] = confidences
    return inputs
