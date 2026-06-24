# AWS Machine Learning Foundations

**Completed labs and demos from the AWS Academy ML Foundations course — the full ML lifecycle run on Amazon SageMaker, Rekognition, and Lex.**

![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white) ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) ![AWS SageMaker](https://img.shields.io/badge/Amazon%20SageMaker-232F3E?style=flat&logo=amazonaws&logoColor=white) ![XGBoost](https://img.shields.io/badge/XGBoost-006600?style=flat&logo=xgboost&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white) ![pandas](https://img.shields.io/badge/pandas-150458?style=flat&logo=pandas&logoColor=white)

## Overview

This repo is my worked-through version of the **AWS Academy Machine Learning Foundations** course. It isn't a single application — it's a set of guided labs and small demos where I built and ran the standard ML workflow on real AWS services: pulling data into S3, training models on Amazon SageMaker, running batch inference, scoring the results, and tuning hyperparameters. There's also a computer-vision lab on Amazon Rekognition and a browser-based chatbot client that talks to an Amazon Lex bot.

The course notebooks ship as scaffolding with the implementation cells left empty. I filled them in and added my own inline comments explaining what each step does and why (for example, on `Flight_Delay-Student.ipynb`: 102 code cells completed, 36 of them with my own annotations). The repo is the trail of that work, and it maps to the AWS Academy ML Foundations badge I earned (verifiable on Credly).

Two datasets carry most of the modeling labs: an orthopedic **Vertebral Column** dataset (abnormal vs. normal classification for a healthcare scenario) and a US **flight on-time performance** dataset (predicting whether a flight is delayed).

## Key Features

- End-to-end SageMaker classification pipeline on the Vertebral Column dataset: import → EDA → split → S3 upload → train → batch transform → evaluate → hyperparameter tuning.
- XGBoost training through SageMaker's built-in container (`xgboost 1.0-1`), wired up with `Estimator`, `TrainingInput` channels, and `model.fit()`.
- Batch inference with SageMaker `Transformer` jobs, reading predictions back from S3 and converting probabilities to class labels with a threshold function.
- Model evaluation done by hand: confusion matrix, sensitivity/specificity, and ROC/AUC curves with scikit-learn and seaborn.
- Hyperparameter tuning jobs (`HyperparameterTuner`) searching over `alpha`, `min_child_weight`, `subsample`, `eta`, and `num_round`, then attaching the best training job and re-scoring it against the baseline.
- A larger flight-delay lab (`Flight_Delay-Student.ipynb`, 214 cells) that runs a full SageMaker `LinearLearner` binary classifier, adds external weather data as features, and then swaps in XGBoost with tuning for comparison.
- Categorical/ordinal feature-encoding lab on the UCI automobile dataset.
- Computer-vision lab using **Amazon Rekognition** (via boto3): create a face collection, detect faces, search a known face, and draw bounding boxes with PIL/skimage.
- An **Amazon Lex** appointment-scheduling chatbot web client (HTML + AWS JavaScript SDK) authenticating through an Amazon Cognito identity pool.
- My own explanatory comments throughout the notebooks — the code is annotated as study notes, not just left as bare lab output.

## How It Works

The repository is organized by course module rather than as one program. The center of gravity is the SageMaker classification workflow, which the `3_x` notebooks build up step by step and the flight-delay notebook repeats at larger scale.

### Data import and exploration

Datasets are pulled in over HTTP (`requests`), unzipped in-memory (`zipfile`, `io`), and parsed — including ARFF files via `scipy.io.arff` for the Vertebral Column data and a `ucimlrepo` fetch as an alternative loader. From there it's standard pandas EDA: `shape`, summary stats, and distribution/correlation plots with matplotlib and seaborn.

### Train / validate / test split and S3 staging

The data is split into three sets with scikit-learn `train_test_split`, stratified on the class label so the abnormal/normal ratio stays consistent:

```python
train, test_and_validate = train_test_split(df, test_size=0.2, random_state=42, stratify=df['class'])
test,  validate          = train_test_split(test_and_validate, test_size=0.5, random_state=42, stratify=test_and_validate['class'])
```

That gives roughly an 80/10/10 split. Each frame is written to CSV (no header, no index — the format the SageMaker XGBoost container expects) and uploaded to S3 with boto3, where the training and validation channels point.

### Training on SageMaker

The built-in XGBoost container is pulled by region and version, then wrapped in a SageMaker `Estimator`:

```python
container = retrieve('xgboost', boto3.Session().region_name, '1.0-1')
xgb_model = sagemaker.estimator.Estimator(container,
                                          sagemaker.get_execution_role(),
                                          sagemaker_session=sagemaker.Session())
xgb_model.fit(inputs=data_channels, logs=False)
```

Training reads the `train` and `validate` channels defined with `sagemaker.inputs.TrainingInput`. The flight-delay notebook does the equivalent with a `LinearLearner` estimator first (`predictor_type='binary_classifier'`, `ml.m4.xlarge`) using `RecordSet` inputs, then moves to XGBoost for comparison.

### Batch inference and scoring

Instead of standing up a live endpoint, the labs use SageMaker batch transform. A `Transformer` job runs predictions over the test set in S3, the output is read back into pandas, and a threshold function turns the raw probabilities into `0`/`1` classes. Evaluation is then computed directly:

- Confusion matrix (`sklearn.metrics.confusion_matrix`, plotted with seaborn).
- Sensitivity and specificity derived from the TN/FP/FN/TP breakdown.
- ROC curve and AUC (`roc_curve`, `auc`, `roc_auc_score`).

### Hyperparameter tuning

The final modeling step launches a SageMaker `HyperparameterTuner` over the XGBoost estimator, with `eval_metric='error@.40'` and continuous/integer ranges for the regularization and learning parameters:

```python
hyperparameter_ranges = {
    'alpha':            ContinuousParameter(0, 100),   # L1 regularization
    'min_child_weight': ContinuousParameter(1, 5),
    'subsample':        ContinuousParameter(0.5, 1),
    'eta':              ContinuousParameter(0.1, 0.3), # learning rate
    'num_round':        IntegerParameter(1, 50),
}
```

After the tuning job finishes, the notebook queries it with `HyperparameterTuningJobAnalytics`, finds the best training job, attaches an estimator to it, runs another batch transform, and compares the tuned model's metrics against the untuned baseline.

### Computer vision with Rekognition

`05-facedetection.ipynb` uses the managed Rekognition API through boto3 — `boto3.client('rekognition')` — rather than training anything locally. It creates a face collection, sends image bytes for face detection, searches for a known face, and uses PIL/skimage/matplotlib to load images and draw bounding boxes around the matches.

### Lex chatbot client

`Lab 6/` is a small front-end for an Amazon Lex bot. `index.html` loads the AWS JavaScript SDK (`aws-sdk-2.633.0`), configures the `us-east-1` region and a Cognito identity pool for credentials, and posts user messages to a Lex appointment-scheduling bot ("NTBBot" / `ScheduleAppointment`), rendering the conversation in a simple chat form. `error.html` is the fallback page.

## Results / Highlights

These are guided labs, so the numbers are the working artifacts rather than benchmark claims:

- Vertebral Column classification trained and tuned end-to-end on SageMaker XGBoost, evaluated with confusion matrix, sensitivity/specificity, and ROC/AUC, with a before/after comparison against the hyperparameter-tuned model.
- Flight-delay capstone: 214-cell notebook, 102 completed code cells, full LinearLearner → XGBoost progression with weather features added and validation AUC reported via `roc_auc_score`.
- Covers the full course breadth — fundamentals, ETL/encoding, model build and evaluation, forecasting concepts, computer vision, NLP services, and generative AI on AWS — across the notebook set (~2.9 MB).
- Maps to the AWS Academy Graduate – Machine Learning Foundations badge (Credly-verified).

## Tech Stack

- **Languages:** Python, HTML/JavaScript
- **ML / data:** scikit-learn, XGBoost, pandas, NumPy, matplotlib, seaborn, scipy, PIL/Pillow, scikit-image, ucimlrepo
- **AWS:** Amazon SageMaker (Estimator, LinearLearner, XGBoost container, TrainingInput, Transformer, HyperparameterTuner, RecordSet), Amazon Rekognition, Amazon Lex, Amazon Cognito, Amazon S3, boto3
- **Tooling:** Jupyter Notebook, AWS CLI, AWS JavaScript SDK

## Getting Started

### Prerequisites

- Python 3.8+ and pip
- Jupyter Notebook
- An AWS account with permissions for SageMaker, S3, and Rekognition (the notebooks were written to run inside Amazon SageMaker notebook instances, which provide the execution role automatically)
- AWS CLI configured locally if running outside SageMaker

### Installation

```bash
git clone https://github.com/DCode-v05/AWS-Machine-Learning-Foundations.git
cd AWS-Machine-Learning-Foundations

# core libraries used across the notebooks
pip install sagemaker boto3 pandas numpy scikit-learn matplotlib seaborn scipy pillow scikit-image ucimlrepo
```

### Running

```bash
jupyter notebook
```

Open a notebook and run the cells top to bottom. The `3_x-machinelearning.ipynb` files are meant to be read in order (3.2 → 3.7) — each re-runs the earlier setup steps before adding its own. For the Lex demo, open `Lab 6/index.html` in a browser (it needs a valid Cognito identity pool and a deployed Lex bot to actually respond).

## Usage

- **SageMaker labs (`3_2`–`3_7`, `Flight_Delay-Student.ipynb`):** run inside a SageMaker notebook instance so `sagemaker.get_execution_role()` resolves. They train real training jobs and run batch transforms, so they incur AWS charges — the tuning cells in particular take roughly 10 minutes and spin up multiple instances.
- **Rekognition lab (`05-facedetection.ipynb`):** needs an image to upload; set your AWS credentials/region first, then run the collection and face-detection cells.
- **Encoding lab (`3_3`):** self-contained — it pulls the UCI automobile dataset and runs locally without AWS.
- **Lex client (`Lab 6/`):** edit `index.html` to point at your own Cognito identity pool ID and bot name before opening it.

## Project Structure

```
AWS-Machine-Learning-Foundations/
├── 3_2-machinelearning.ipynb        # Vertebral Column: data import + EDA
├── 3_3-machinelearning.ipynb        # Ordinal/categorical encoding (UCI automobile dataset)
├── 3_4-machinelearning.ipynb        # Train/validate/test split, S3 upload, SageMaker XGBoost training
├── 3_5-machinelearning.ipynb        # Model deployment / batch transform
├── 3_6-machinelearning.ipynb        # Evaluation: confusion matrix, sensitivity/specificity, ROC/AUC
├── 3_7-machinelearning.ipynb        # Hyperparameter tuning + best-model comparison
├── Flight_Delay-Student.ipynb       # Capstone: flight-delay prediction (LinearLearner → XGBoost + tuning)
├── 05-facedetection.ipynb           # Computer vision with Amazon Rekognition (boto3)
├── Vertebral Column.ipynb           # Dataset loader (ARFF reader / ucimlrepo)
├── Lab 6/
│   ├── index.html                   # Amazon Lex appointment-bot web client (AWS JS SDK + Cognito)
│   └── error.html                   # Fallback page for the chatbot client
└── README.md
```

---

## Contact

**Portfolio:** [Denistan](https://www.denistan.me)<br>
**LinkedIn:** [Denistan](https://www.linkedin.com/in/denistanb)<br>
**GitHub:** [DCode-v05](https://github.com/DCode-v05)<br>
**LeetCode:** [Denistan_B](https://leetcode.com/u/Denistan_B)<br>
**Email:** [denistanb05@gmail.com](mailto:denistanb05@gmail.com)

Made with ❤️ by **Denistan B**
