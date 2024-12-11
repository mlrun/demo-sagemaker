import mlrun
from kfp import dsl


@dsl.pipeline(
    name="Classify payment Pipeline",
    description="Classify payment from a transactions dataset",
)
def kfpipeline():
    """
    This function defines a Kubeflow Pipeline for payment classification.
    It trains a model using SageMaker, evaluates its performance, sets up a serving function,
    and deploys the serving function.
    """
    project = mlrun.get_current_project()

    # Train
    train_run = project.run_function(
        function="train",
        name="train",
        handler="train",
        params={},
        outputs=["model_path", "model", "test_data"],
    )

    # Evaluate
    project.run_function(
        function="evaluate",
        name="evaluate",
        handler="evaluate",
        inputs={"test_set": train_run.outputs["test_data"]},
        params={
            "model_path": train_run.outputs["model_path"],
            "model_name": "xgboost-model",
            "label_column": "transaction_category",
        },
        returns=["classification_report: dataset"],
    )

    serving_function = project.get_function("serving")

    if serving_function.spec.graph is None:
        # Set the topology and get the graph object:
        graph = serving_function.set_topology("flow", engine="async")

        # Add the steps:
        graph.to("XGBModelServer", name="xgboost-model", model_path=project.get_artifact('train_my-model').target_path).to(
            handler="postprocess", name="postprocess"
        ).respond()

    # Deploy the serving function:
    project.deploy_function(serving_function).after(train_run)
