import os

import boto3
import mlrun.feature_store as fs
import numpy as np
import sagemaker


def train(context):
    # Set AWS environment variables:
    _set_envars(context)

    # Get data from feature-store:
    data = _get_feature_store_data(context)

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
    role = context.get_secret("SAGEMAKER-ROLE")
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
    os.environ["SAGEMAKER-ROLE"] = context.get_secret("SAGEMAKER-ROLE")


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
