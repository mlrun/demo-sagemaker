# Copyright 2023 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import mlrun

ENV_FILE_PATH = "mlrun.env"

def setup(
    project: mlrun.projects.MlrunProject,
) -> mlrun.projects.MlrunProject:
    """
    Creating the project for the demo. This function is expected to call automatically when calling the function
    `mlrun.get_or_create_project`.

    :param project: The project to set up.

    :returns: A fully prepared project for this demo.
    """
    # Set secrets:
    _set_secrets(project=project)

    # Unpack parameters:
    source = project.get_param(key="source")
    default_image = project.get_param(key="default_image")

    # Set or build the default image:
    if default_image is None and not project.spec.build.image:
        print("Building default image for the demo:")
        _build_image(project=project)
    else:
        project.set_default_image(default_image)

    # Set the project git source:
    if source:
        print(f"Project Source: {source}")
        project.set_source(source=source, pull_at_runtime=True)

    # Set the functions:
    _set_functions(project=project)

    # Set the workflows:
    project.set_workflow(name="sagemaker", workflow_path="./src/workflows/workflow.py")

    # Save and return the project:
    project.save()
    return project


def _set_secrets(project: mlrun.projects.MlrunProject):
    # Set the secrets:
    project.set_secrets(file_path=ENV_FILE_PATH)
    # Set as environment variables:
    mlrun.set_env_from_file(ENV_FILE_PATH)


def _build_image(project: mlrun.projects.MlrunProject):
    assert project.build_image(
        base_image="mlrun/mlrun",
        commands=[
            "pip install sagemaker",
            "pip install xgboost",
        ],
        set_as_default=True,
    )


def _set_functions(project: mlrun.projects.MlrunProject):
    # Train
    _set_function(
        project=project,
        func="src/functions/train.py",
        name="train",
        kind="job",
    )

    # Evaluate
    _set_function(
        project=project,
        func="src/functions/evaluate.py",
        name="evaluate",
        kind="job",
    )

    # Serving
    _set_function(
        project=project,
        func="src/functions/data-preparation.py",
        name="data-preparation",
        kind="job",
    )

    # Serving
    _set_function(
        project=project,
        func="src/functions/serving.py",
        name="serving",
        kind="serving",
    )


def _set_function(
    project: mlrun.projects.MlrunProject,
    func: str,
    name: str,
    kind: str,
):
    # Set the given function:
    mlrun_function = project.set_function(
        func=func,
        name=name,
        kind=kind,
    )

    # Save:
    mlrun_function.save()
