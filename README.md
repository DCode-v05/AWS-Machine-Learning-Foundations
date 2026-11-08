# AWS Machine Learning Foundations

Notebooks and a small web demo from the AWS Academy Machine Learning Foundations course. The labs cover the usual ML workflow on Amazon SageMaker, face detection with Rekognition, and a chatbot client for Amazon Lex.

The course notebooks come with the implementation cells empty. These are my filled-in versions. It is a set of labs, not one application.

## What is in here

| File | What it does |
|---|---|
| `Vertebral Column.ipynb` | Two ways of loading the Vertebral Column dataset (ARFF via `scipy.io.arff`, and `ucimlrepo`) |
| `3_2-machinelearning.ipynb` | Data import and exploration of the Vertebral Column data (normal vs. abnormal) |
| `3_3-machinelearning.ipynb` | Ordinal and categorical encoding on the UCI automobile dataset (runs locally, no AWS needed) |
| `3_4-machinelearning.ipynb` | Stratified train/validation/test split, upload to S3, XGBoost training on SageMaker |
| `3_5-machinelearning.ipynb` | Hosting the trained model and running predictions |
| `3_6-machinelearning.ipynb` | Evaluation: confusion matrix, sensitivity/specificity, ROC and AUC |
| `3_7-machinelearning.ipynb` | Hyperparameter tuning with `HyperparameterTuner`, then comparing the tuned model with the baseline |
| `Flight_Delay-Student.ipynb` | Flight delay prediction: SageMaker `LinearLearner` first, then XGBoost with tuning |
| `05-facedetection.ipynb` | Rekognition through boto3: create a collection, add and search faces, draw bounding boxes |
| `Lab 6/` | `index.html` and `error.html`, a web client for an Amazon Lex appointment bot (AWS JavaScript SDK, Cognito identity pool) |

## Setup

The SageMaker notebooks expect to run inside a SageMaker notebook instance, where `sagemaker.get_execution_role()` resolves on its own. Outside SageMaker you need AWS credentials and an execution role yourself.

```bash
pip install sagemaker boto3 pandas numpy scikit-learn matplotlib seaborn scipy pillow scikit-image ucimlrepo
jupyter notebook
```

Run the `3_x` notebooks in order. Each one repeats the earlier setup cells before adding its own step.

The SageMaker training and tuning jobs cost money. The notebooks note that the tuning cells take around 10 to 45 minutes.

For the Lex client, open `Lab 6/index.html` in a browser. It points at a Cognito identity pool and a bot named `ScheduleAppointment` in `us-east-1`, so you need to change those to your own.

## Tech stack

Python, scikit-learn, XGBoost, pandas, matplotlib, seaborn, Amazon SageMaker, Rekognition, Lex, Cognito, S3, boto3, AWS SDK for JavaScript.
