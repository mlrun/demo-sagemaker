import os

import boto3
import mlrun.feature_store as fs
import numpy as np
import sagemaker
import pandas as pd
from sagemaker.feature_store.feature_group import FeatureGroup
import time




def train(context):
    # Set AWS environment variables:
    _set_envars(context)


    region = sagemaker.Session().boto_region_name
    sm_client = boto3.client("sagemaker")
    boto_session = boto3.Session(region_name=region)
    sagemaker_session = sagemaker.session.Session(boto_session=boto_session, sagemaker_client=sm_client)
    role = os.environ["SAGEMAKER_ROLE"]
    bucket_prefix = "payment-classification"
    s3_bucket = sagemaker_session.default_bucket()

    factorize_key = {
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

    factorize_key = {key: str(value) for key, value in factorize_key.items()}

    s3 = boto3.client("s3")
    s3.download_file(
        f"sagemaker-example-files-prod-{region}",
        "datasets/tabular/synthetic_financial/financial_transactions_mini.csv",
        "financial_transactions_mini.csv",
    )

    data = pd.read_csv(
        "financial_transactions_mini.csv",
        parse_dates=["timestamp"],
        infer_datetime_format=True,
        dtype={"transaction_category": "string"},
    )

    data["year"] = data["timestamp"].dt.year
    data["month"] = data["timestamp"].dt.month
    data["day"] = data["timestamp"].dt.day
    data["hour"] = data["timestamp"].dt.hour
    data["minute"] = data["timestamp"].dt.minute
    data["second"] = data["timestamp"].dt.second

    del data["timestamp"]

    data["transaction_category"] = data["transaction_category"].replace(factorize_key)

    feature_group_name = "feature-group-payment-classification"
    record_identifier_feature_name = "identifier"

    feature_group = FeatureGroup(name=feature_group_name, sagemaker_session=sagemaker_session)

    featurestore_runtime = boto_session.client(
        service_name="sagemaker-featurestore-runtime", region_name=region
    )

    feature_store_session = sagemaker.Session(
        boto_session=boto_session,
        sagemaker_client=sm_client,
        sagemaker_featurestore_runtime_client=featurestore_runtime,
    )

    columns = ["mean_amount", "count", "identifier", "EventTime"]
    feature_store_data = pd.DataFrame(columns=columns, dtype=object)

    feature_store_data["identifier"] = range(19)
    feature_store_data["mean_amount"] = 0.0
    feature_store_data["count"] = 1
    feature_store_data["EventTime"] = time.time()

    feature_group.load_feature_definitions(data_frame=feature_store_data)

    status = feature_group.describe().get("FeatureGroupStatus")

    if status!='Created':
        feature_group.create(
            s3_uri=f"s3://{s3_bucket}/{bucket_prefix}",
            record_identifier_name=record_identifier_feature_name,
            event_time_feature_name="EventTime",
            role_arn=role,
            enable_online_store=True,
        )

    status = feature_group.describe().get("FeatureGroupStatus")
    while status == "Creating":
        print("Waiting for Feature Group to be Created")
        time.sleep(5)
        status = feature_group.describe().get("FeatureGroupStatus")
    print(f"FeatureGroup {feature_group.name} successfully created.")

    feature_group.ingest(data_frame=feature_store_data, max_workers=3, wait=True)

    def get_feature_store_values():
        response = featurestore_runtime.batch_get_record(
            Identifiers=[
                {
                    "FeatureGroupName": feature_group_name,
                    "RecordIdentifiersValueAsString": [str(i) for i in range(19)],
                }
            ]
        )

        columns = ["mean_amount", "count", "identifier", "EventTime"]

        feature_store_resp = pd.DataFrame(
            data=[
                [resp["Record"][i]["ValueAsString"] for i in range(len(columns))]
                for resp in response["Records"]
            ],
            columns=columns,
        )
        feature_store_resp["identifier"] = feature_store_resp["identifier"].astype(int)
        feature_store_resp["count"] = feature_store_resp["count"].astype(int)
        feature_store_resp["mean_amount"] = feature_store_resp["mean_amount"].astype(float)
        feature_store_resp["EventTime"] = feature_store_resp["EventTime"].astype(float)
        feature_store_resp = feature_store_resp.sort_values(by="identifier")

        return feature_store_resp

    feature_store_resp = get_feature_store_values()

    feature_store_data = pd.DataFrame()
    feature_store_data["mean_amount"] = data.groupby(["transaction_category"]).mean()["amount"]
    feature_store_data["count"] = data.groupby(["transaction_category"]).count()["amount"]
    feature_store_data["identifier"] = feature_store_data.index
    feature_store_data["EventTime"] = time.time()

    feature_store_data["mean_amount"] = (
        pd.concat([feature_store_resp, feature_store_data])
        .groupby("identifier")
        .apply(lambda x: np.average(x["mean_amount"], weights=x["count"]))
    )
    feature_store_data["count"] = (
        pd.concat([feature_store_resp, feature_store_data]).groupby("identifier").sum()["count"]
    )

    feature_group.ingest(data_frame=feature_store_data, max_workers=3, wait=True)

    feature_store_data = get_feature_store_values()

    additional_features = pd.pivot_table(
        feature_store_data, values=["mean_amount"], index=["identifier"]
    ).T.add_suffix("_dist")
    additional_features_columns = list(additional_features.columns)
    data = pd.concat([data, pd.DataFrame(columns=additional_features_columns, dtype=object)])
    data[additional_features_columns] = additional_features.values[0]
    for col in additional_features_columns:
        data[col] = abs(data[col] - data["amount"])

    # Randomly sort the data then split out first 70%, second 20%, and last 10%
    train_data, validation_data, test_data = np.split(
        data.sample(frac=1, random_state=42),
        [int(0.7 * len(data)), int(0.9 * len(data))],
    )

    # Save the data locally:
    train_data.to_csv("train.csv", index=False, header=False)
    validation_data.to_csv("validation.csv", index=False, header=False)
    test_data.to_csv("test.csv", index=False, header=True)

    # Setting up a session:
    region = sagemaker.Session().boto_region_name
    sm_client = boto3.client("sagemaker")
    boto_session = boto3.Session(region_name=region)
    sagemaker_session = sagemaker.session.Session(
        boto_session=boto_session, sagemaker_client=sm_client
    )
    role = context.get_secret("SAGEMAKER_ROLE")
    bucket_prefix = "payment-classification"
    s3_bucket = sagemaker_session.default_bucket()

    # Uploading csv files to s3:
    boto3.Session().resource("s3").Bucket(s3_bucket).Object(
        os.path.join(bucket_prefix, "train/train.csv")
    ).upload_file("train.csv")
    boto3.Session().resource("s3").Bucket(s3_bucket).Object(
        os.path.join(bucket_prefix, "validation/validation.csv")
    ).upload_file("validation.csv")
    boto3.Session().resource("s3").Bucket(s3_bucket).Object(
        os.path.join(bucket_prefix, "test/test.csv")
    ).upload_file("test.csv")

    # Retrieve container from sagemaker:
    container = sagemaker.image_uris.retrieve(
        region=region, framework="xgboost", version="1.2-2"
    )

    # Convert train and validation datasets into TrainingInput objects:
    s3_input_train = sagemaker.inputs.TrainingInput(
        s3_data="s3://{}/{}/train".format(s3_bucket, bucket_prefix), content_type="csv"
    )
    s3_input_validation = sagemaker.inputs.TrainingInput(
        s3_data="s3://{}/{}/validation/".format(s3_bucket, bucket_prefix),
        content_type="csv",
    )

    # Set up an xgb estimator:
    xgb = sagemaker.estimator.Estimator(
        container,
        role,
        instance_count=1,
        instance_type="ml.m4.xlarge",
        output_path="s3://{}/{}/output".format(s3_bucket, bucket_prefix),
        sagemaker_session=sagemaker_session,
    )

    # Set up hyperparameters:
    xgb.set_hyperparameters(
        max_depth=5,
        eta=0.2,
        gamma=4,
        min_child_weight=6,
        subsample=0.8,
        objective="multi:softprob",
        num_class=19,
        verbosity=0,
        num_round=100,
    )

    # Apply training:
    xgb.fit({"train": s3_input_train, "validation": s3_input_validation}, logs="None")

    # Save the model's path:
    context.log_artifact("model_path", body=xgb.model_data)

    # Save the test data path:
    context.log_artifact(
        "test_data", body="s3://{}/{}/test/test.csv".format(s3_bucket, bucket_prefix)
    )


def _set_envars(context):
    os.environ["AWS_ACCESS_KEY_ID"] = context.get_secret("AWS_ACCESS_KEY_ID")
    os.environ["AWS_SECRET_ACCESS_KEY"] = context.get_secret("AWS_SECRET_ACCESS_KEY")
    os.environ["AWS_DEFAULT_REGION"] = context.get_secret("AWS_DEFAULT_REGION")
    os.environ["SAGEMAKER_ROLE"] = context.get_secret("SAGEMAKER_ROLE")


def _get_feature_store_data(context):
    project_name = context.project
    features = [
        f"{project_name}/transactions.*",
    ]

    vector = fs.FeatureVector(
        "transactions", features=features, description="enriched transactions"
    )
    resp = fs.FeatureVector.get_offline_features(vector)

    # Preview the dataset
    df = resp.to_dataframe()

    return df
