# AWS SageMaker Demo with MLRun

This demo showcases how to build, manage, and deploy machine learning models using AWS SageMaker and MLRun. It emphasizes the automation of ML workflows from development to production.

This demo is based on the SageMaker Payment Classification use case from the SageMaker's example repository (https://github.com/aws/amazon-sagemaker-examples/blob/main/use-cases/financial_payment_classification/financial_payment_classification.ipynb). 

## Key Components

- **AWS SageMaker**: A comprehensive service that enables developers and data scientists to build, train, and deploy machine learning (ML) models efficiently.

- **MLRun**: An open-source MLOps framework designed to manage and automate your machine learning and data science lifecycle. In this demo, it is used to automate ML deployment and workflows.

## Running the Demo

1. **Prerequisites**: Ensure you have an AWS account with SageMaker enabled and MLRun installed in your environment.

2. **Clone the repository**: Clone this repository to your SageMaker notebook environment.

3. **Set the environment variables in `mlrun.env`**: Copy the `mlrun.env` file to your workspace and fill in the necessary environment variables such as `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`, `SAGEMAKER_ROLE`, `MLRUN_DBPATH`, `V3IO_USERNAME`, and `V3IO_ACCESS_KEY`.

4. **Run the Jupyter notebook**: Open and run the `financial-payment-pipeline.ipynb` notebook. This notebook contains the code for the financial payment classification pipeline.

5. **Monitor your runs**: Track your runs in the MLRun dashboard. The dashboard provides a graphical interface for tracking your MLRun projects, functions, runs, and artifacts.

You can also open `financial-payment-classification.ipynb` to review the SageMaker code and the MLRun code segments cell-by-cell. This notebook does not include the automated workflow, but rather the individual steps.

## CI/CD using GitHub Actions
This demo also includes a workflow for automating the execution of the machine learning pipeline. To set this up:

1. **Fork this repository**: Create a copy of this repository in your own GitHub account by forking it.

2. **Add Secrets to Your Repository**: Navigate to the "Settings" tab in your GitHub repository, then click on "Secrets". Here, you need to add the following secrets, which will be used as environment variables in your workflow:

- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_DEFAULT_REGION`
- `SAGEMAKER_ROLE`
- `MLRUN_DBPATH`
- `V3IO_ACCESS_KEY`

Additionally, set the `V3IO_USERNAME` environment variable to your username. 

3. **Commit and Push Your Changes**: Make any necessary changes to the code, then commit and push these changes to your repository. 

4. **Create a Pull Request**: Create a pull request to either the `staging` or `main` branch. Once the pull request is merged, it will trigger the GitHub action. You can review the pipeline execution in the MLRun UI, a link to which can be found in the workflow steps.

You can also run the workflow manually by navigating to the "Actions" tab in your repository and clicking on the workflow.

## License

This demo is licensed under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0). For more details, please take a look at the [LICENSE](./LICENSE) file.
