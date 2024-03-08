import click
import mlrun

@click.command()
@click.option(
    "--project-name",
    required=True,
    help="Specify the project name.",
)
@click.option(
    "--workflow-name",
    required=True,
    help="Specify the workflow name.",
)
@click.option(
    "--repo",
    required=True,
    help="The repo name.",
    default="archive",
)
@click.option(
    "--branch",
    type=click.Choice(["development", "staging", "master"]
),
    required=True,
    help="Specify the branch - only relevant when using git source.",
    default="development",
)
@click.option(
    "--single-cluster-mode",
    is_flag=True,
    help="Specify whether the environments exist in the same cluster.",
    default=False,
)
def main(
    project_name: str, workflow_name: str, repo: str, branch: str, single_cluster_mode: bool
) -> None:

    user_project = (
        True if single_cluster_mode and branch in ["staging", "master"] else False
    )

    source = f"{repo}#{branch}"

    print(f"Loading project {project_name} with source {source}")

    project = mlrun.get_or_create_project(
        name=project_name, 
        user_project=user_project,
        parameters={
            "source" : source,
            "default_image" : "yonishelach/sagemaker-demo"
        }
    )

    print(f"Running workflow {workflow_name}...")
    project.run(name=workflow_name, dirty=True, watch=True)


if __name__ == "__main__":
    main()
