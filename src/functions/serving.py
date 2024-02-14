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