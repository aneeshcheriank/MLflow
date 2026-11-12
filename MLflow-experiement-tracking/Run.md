## Run the experiment
- Need to have 
    - MLproject
    - yaml type file without .yaml extention
    - structure

```yaml
name: learning_rate_fine-tuning

entry_points:
    main:
        parameters:
            channels: {type: int, default: 8}
            window: {type: int, default: 3}
            lr: {type: float, default: 0.001}
        command: "python train.py --channels {channels} --window {window} --lr {lr}"
```

    - train.py file
        - command line arguments
            - click options
            - mlflow.tensorflow.autolog()

- run commnad `mlflow run . --run-name learning_rate -P lr 0.001 --env-manager local`
    - env-manaer can be ['local', 'conda', 'virtualenv', 'uv']
    - insted of . which represent the URI, we can give a git location if we want