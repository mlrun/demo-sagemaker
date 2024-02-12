import tarfile
import warnings
from typing import List

import mlrun
import numpy as np
import xgboost as xgb
from cloudpickle import load
import mlrun.feature_store as fstore


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

        print(body)
        # body['inputs'][0] = body['inputs'][0][1:]

        # print(body)



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

# def get_realtime_transactions_aggregations():
#     # Create a feature vector that gets the average amount
#     vector = fstore.FeatureVector("aggregations-vector", ["aggregations.amount_avg_1d"], with_indexes=True)
#     #get the categories list
#     unique_categories = ["0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16"]
#     # Use online feature service to get the latest average amount per category
#     with vector.get_online_feature_service() as online_feature_service:
#         resp = online_feature_service.get(
#             [{"transaction_category":cat} for cat in unique_categories]
#         )
#     return resp

# def calculate_distances(resp, event):
#     for cat in resp:
#         transaction_category = cat['transaction_category']        
#         amount_avg = cat['amount_avg_1d']
#         event[0]["dist_" + transaction_category] = abs(amount_avg - event[0]["amount"])

#     return event

# def convert_timestamp_to_components(event):
#     event[0]["year"] = event[0]["timestamp"].year
#     event[0]["month"] = event[0]["timestamp"].month
#     event[0]["day"] = event[0]["timestamp"].day
#     event[0]["hour"] = event[0]["timestamp"].hour
#     event[0]["minute"] = event[0]["timestamp"].minute
#     event[0]["second"] = event[0]["timestamp"].second
#     del event[0]['timestamp']

#     return event

# def move_to_end(ls, key):
#     """Move an item to the end of the dictionary."""
#     d = ls[0]
#     if key in d:
#         value = d.pop(key)  # Remove the item and get its value
#         d[key] = value  # Reinsert the item, which moves it to the end
#     ls[0] = d
#     return ls




# # Function that preprocesses the inference data
# def preprocess(event):    
#     resp = get_realtime_transactions_aggregations()
#     dist_event = calculate_distances(resp, event)
#     converted_event = convert_timestamp_to_components(event)
#     restructured_event = move_to_end(converted_event,'transaction_id')
#     values_list = list(restructured_event[0].values())
#     return_list = [values_list]
#     return_list
#     return_dict = {"inputs": return_list}
#     return return_dict
    