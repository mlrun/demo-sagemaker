import boto3
import io
import sagemaker
import time
import os

from time import sleep
from sklearn.metrics import classification_report
#from sagemaker.feature_store.feature_group import FeatureGroup

import pandas as pd
import numpy as np

import numpy as np
import pandas as pd
import mlrun.feature_store as fs


def get_feature_store_data():        
    features = [
        "sagemaker-payment/transactions.*",
    ]

    vector = fs.FeatureVector("extended-transactions",features=features,description="enriched transactions")
    resp = fs.FeatureVector.get_offline_features(vector)

    #Preview the dataset
    df = resp.to_dataframe()
    
    return df

def save_data(train_data, validation_data, test_data):
    train_data.to_csv("train.csv", index=False, header=False)
    validation_data.to_csv("validation.csv", index=False, header=False)
    test_data.to_csv("test.csv", index=False, header=False)
    
    
def upload_files():
    boto3.Session().resource("s3").Bucket(s3_bucket).Object(
    os.path.join(bucket_prefix, "train/train.csv")
    ).upload_file("train.csv")
    boto3.Session().resource("s3").Bucket(s3_bucket).Object(
    os.path.join(bucket_prefix, "validation/validation.csv")
    ).upload_file("validation.csv")

def train(context):
    data = get_feature_store_data()         
    sagemaker_role = context.get_secret("SAGEMAKER-ROLE")
    context.logger.info(sagemaker_role)
    # Randomly sort the data then split out first 70%, second 20%, and last 10%
    train_data, validation_data, test_data = np.split(
         data.sample(frac=1, random_state=42), [int(0.7 * len(data)), int(0.9 * len(data))]
    )
    #save_data(train_data, validation_data, test_data)    
    #upload_files()
    train_data.to_csv("train.csv", index=False, header=False)
    validation_data.to_csv("validation.csv", index=False, header=False)
    test_data.to_csv("test.csv", index=False, header=False)
    
    region = sagemaker.Session().boto_region_name
    sm_client = boto3.client("sagemaker")
    boto_session = boto3.Session(region_name=region)
    sagemaker_session = sagemaker.session.Session(boto_session=boto_session, sagemaker_client=sm_client)
    #role = sagemaker.get_execution_role()
    role = sagemaker_role
    bucket_prefix = "payment-classification"
    s3_bucket = sagemaker_session.default_bucket()
    
    boto3.Session().resource("s3").Bucket(s3_bucket).Object(
    os.path.join(bucket_prefix, "train/train.csv")
    ).upload_file("train.csv")
    boto3.Session().resource("s3").Bucket(s3_bucket).Object(
    os.path.join(bucket_prefix, "validation/validation.csv")
    ).upload_file("validation.csv")
    

    container = sagemaker.image_uris.retrieve(region=region, framework="xgboost", version="1.2-2")
    
    s3_input_train = sagemaker.inputs.TrainingInput(
    s3_data="s3://{}/{}/train".format(s3_bucket, bucket_prefix), content_type="csv")
    s3_input_validation = sagemaker.inputs.TrainingInput(
    s3_data="s3://{}/{}/validation/".format(s3_bucket, bucket_prefix), content_type="csv")
    
    xgb = sagemaker.estimator.Estimator(
    container,
    role,
    instance_count=1,
    instance_type="ml.m4.xlarge",
    output_path="s3://{}/{}/output".format(s3_bucket, bucket_prefix),
    sagemaker_session=sagemaker_session,)
    
    xgb.set_hyperparameters(
    max_depth=5,
    eta=0.2,
    gamma=4,
    min_child_weight=6,
    subsample=0.8,
    objective="multi:softprob",
    num_class=19,
    verbosity=0,
    num_round=100,)
    
    xgb.fit({"train": s3_input_train, "validation": s3_input_validation},logs="None")
    

    
    context.log_artifact("model_path", body=xgb.model_data)
    
