import tarfile
import tempfile
from typing import Union

import mlrun
import pandas as pd
import xgboost as xgb
from cloudpickle import load
from sklearn.metrics import classification_report

FACTORIZE_KEY = {
    "Uncategorized": 0,
    "Entertainment": 1,
    "Education": 2,
    "Shopping": 3,
    "Personal Care": 4,
    "Health and Fitness": 5,
    "Food and Dining": 6,
    "Gifts and Donations": 7,
    "Investments": 8,
    "Bills and Utilities": 9,
    "Auto and Transport": 10,
    "Travel": 11,
    "Fees and Charges": 12,
    "Business Services": 13,
    "Personal Services": 14,
    "Taxes": 15,
    "Gambling": 16,
    "Home": 17,
    "Pension and insurances": 18,
}


def evaluate(
    model_path: str,
    model_name: str,
    test_set: Union[mlrun.DataItem, str],
    label_column: str,
    factorize_key: dict = None,
) -> pd.DataFrame:
    # download model from s3:
    if model_path.startswith("store://"):
        model_path = mlrun.get_dataitem(model_path).get().decode("utf-8")
    model_temp_path = _download_object_from_s3(model_path, suffix=".tar.gz")

    # extract model file:
    t = tarfile.open(model_temp_path, "r:gz")
    t.extractall()

    # load model from file:
    model = load(open(model_name, "rb"))

    # load data from s3:
    test_set_temp_path = _download_object_from_s3(test_set, suffix=".csv")

    # convert to pandas dataframe:
    test_set = pd.read_csv(test_set_temp_path)

    print(test_set)

    # convert to xgboost object:
    test_data = xgb.DMatrix(test_set.drop(columns=[label_column], axis=1))

    # get predictions:
    predictions = [prediction.argmax() for prediction in model.predict(test_data)]

    if factorize_key is None:
        factorize_key = FACTORIZE_KEY

    # generate classification report:
    report = classification_report(
        y_true=test_set["transaction_category"].to_list(),
        y_pred=predictions,
        target_names=factorize_key,
        output_dict=True,
    )
    report_df = pd.DataFrame.from_dict(report).transpose()

    return report_df


def _download_object_from_s3(object_path: Union[mlrun.DataItem, str], suffix: str):
    if isinstance(object_path, mlrun.DataItem):
        object_path = object_path.get().decode("utf-8")
    obj = mlrun.datastore.store_manager.object(url=object_path)
    temp_path = tempfile.NamedTemporaryFile(
        suffix=suffix,
        delete=False,
    ).name
    obj.download(temp_path)
    return temp_path
