# MLflow
Brush up MLflow

## Key Terms
- MLflow Tracking
    - Logs metrics, parametes, models and other artifacts when runnign ML code to monitor experiments
- MLflow Projects
    - Configurable standared format for organizing ML code to ensure consistency and reprooducibility
- MLflow Models
    - Package ML model files with their dependencis so they can be deployed on diverse platforms


## MLflow tracking
- Logs
    - parameters, metrics, artifacts
    - mlflow.log_metrics("accuracy", 0.45)
    - the logs are the key value pairs

    ```python
    import mlflow
    # parameters
    mlflow.log_param("mum_dimentions", 8)

    # metrics
    mlflow.log_metrics("accuracy", 0.1)

    # artifacts
    mlflow.log_artifacts("roc.png")
    ```

## MLflow Projects
- Packaging data science codee
- a directory with code of a Git repository
- descriptor file to specify dependencies
- YAML file called MLprojects

```yaml
name: My Projct
conda_env: conda.yaml
entry_points:
    main:
        parameters:
            data_file: path
            regulariyation: {type: float, default: 0.1}
        command: "python train.py -r {regularization} {data_file}"
    validate:
        parameters:
            data_file: path
            command: "python validate.py {data_file}"
```

## MLflow Models
- packing models in multiple formats called "flavors"
```txt
time_created:
flavors:
    sklearn:
        sklearn_version: 0.19.1
        pickled_model: model.pkl
    python_function:
        loder_module:   mlflow.sklearn
        pickled_model: model.pkl
```

## Installing mlflow
    - `pip install mlflow`
    - run mlflow `mlflow ui`

## Real-Life Application: Managing Model Experiments as a Data Scientist

- As a Data Scientist, you are likely constantly iterating on models to improve performance. MLflow Tracking is the essential tool for managing these iterations effectively.

    - Scenario: Imagine you are tasked with optimizing a machine learning model to predict customer churn. You decide to test three different regularization values (parameters) to see which yields the highest accuracy (metric) while ensuring your training dataset and final model binary (artifacts) are preserved for audit purposes.
    - Why this is relevant: Without a tracking system, you might lose track of which combination of settings produced your best result, or worse, struggle to recreate a specific model version later. Using MLflow ensures that your development process is rigorous, reproducible, and transparent.

- Breaking down the connection:

    - Logging Parameters: You log your regularization values (e.g., 0.01, 0.1, 1.0). This keeps your "experimental recipe" documented, ensuring you know exactly what configuration led to a specific model iteration.
    - Logging Metrics: By logging accuracy or F1-score for each run, you can use the MLflow UI to visualize which parameters are pushing your model toward better performance, allowing you to make data-driven decisions on where to focus your tuning efforts next.
    - Logging Artifacts: You save the specific version of your dataset and the resulting model binary. If a stakeholder asks why a model is performing a certain way six months from now, you can retrieve the exact artifact associated with that experiment.
    - Version Control: By maintaining this record, you ensure that every change you make to your code or configuration is accounted for, preventing the common "it worked on my machine" issue and making your workflow much more professional and scalable.

By treating your experiments this way, you move from manual, disorganized testing to a professional MLOps workflow where every experiment adds value to your knowledge base.

## Tracking UI
- MLflow experiement 
    - `mlfow experiements create --experiement-name <experiment-name>`
    - return an id
    - now need to collect merrics for the experiemnt
    - `MLFLOW_EXPERIMENT_ID=2 python produce-metrics.py`
    - `tree mlruns/`

## Parameters, Version, Artifacts, and Metrics

# MLflow Projects

## Key terms
- Conda Environment - Specifies the dependencies and software required to recreate the runtime environment
- Entry Points - Define the scripts that can be executed within the project workflow
- Git Repository - Remote repository that contains an MLflow project enabling portability
- mlflow run - Executes an MLflow project locally or from a Git repo with given parameters

## Summary

- MLflow Projects are a format for packaging data science code to run it in a reproducible and reusable way. They consist of code, its dependencies, and metadata.
- Projects can specify details like name, environments, parameters, and entry points in a MLproject file. Otherwise, conventions are used to infer these. 
- Projects can be run locally or remotely using mlflow run. Additional parameters, arguments, and flags can configure the runs.
- MLflow Projects make iterating quickly easier. They also enable constructing multi-step workflows by calling mlflow.projects.run() in a program.
- Use cases include modularizing code, hyperparameter tuning, cross-validation, and packaging existing code.