# AWS Machine Learning Foundations – Learning Journey

## Description
This repository documents my hands-on learning journey through the **AWS Machine Learning Foundations** course by AWS Academy. It covers the end-to-end machine learning lifecycle using real AWS tools and services. Here, you'll find practical notebooks, project labs, and web demos that showcase how to build, train, and deploy machine learning models in the AWS cloud.

Whether you're new to machine learning or looking to understand how ML is implemented on AWS, this repository provides step-by-step guides, code samples, and project-based learning.

## Learning Modules

### 1. Machine Learning Fundamentals
Gain a foundational understanding of what machine learning is, the types of problems it can solve, and the typical workflow of an ML project. Learn about the differences between supervised, unsupervised, and reinforcement learning, and explore real-world business applications of ML. Introduction to AWS services like Amazon SageMaker for ML development.

### 2. Working with Data
Learn how to formulate ML problems from business goals, collect and transform data (ETL), and ensure data security. Explore techniques for data cleaning, feature engineering, and handling outliers. Practice encoding categorical features and preparing datasets for modeling using Jupyter Notebooks and AWS tools.

### 3. Model Building and Evaluation
Understand the process of training, validating, and deploying machine learning models using Amazon SageMaker. Learn about model evaluation metrics such as accuracy, precision, recall, and F1-score. Explore hyperparameter tuning and how to optimize models for better performance. Deploy models and generate predictions using AWS infrastructure.

### 4. Time Series Forecasting
Dive into time series data analysis and forecasting using Amazon Forecast. Learn about time series fundamentals, preprocessing, and how to build automated forecasts for business scenarios. Practice building and evaluating forecasting models with hands-on labs.

### 5. Computer Vision
Explore image and video analysis using AWS services like Amazon Rekognition. Learn how to perform face detection, object labeling, and build custom datasets for computer vision tasks. Understand the basics of data labeling with Amazon Ground Truth and how to use pre-trained models for visual recognition.

### 6. Natural Language Processing (NLP)
Discover the basics of NLP and its applications in real-world scenarios. Use AWS managed services such as Amazon Comprehend, Polly, and Translate to analyze, synthesize, and translate text. Gain hands-on experience with sentiment analysis, entity recognition, and language translation.

### 7. Generative AI with AWS
Get introduced to generative AI concepts, including prompt engineering and foundation models (FMs) in AWS. Learn how to use Amazon Q Developer to create and deploy generative AI models, and explore the latest advancements in AI-driven content creation.

## Tech Stack
- **Languages:** Python, HTML
- **Cloud:** AWS (SageMaker, Rekognition, Forecast, Comprehend, S3)
- **Tools:** Jupyter Notebook, AWS CLI
- **Libraries:** pandas, numpy, scikit-learn, XGBoost, matplotlib

## Getting Started

### Prerequisites
- Python 3.8+
- pip (Python package manager)
- AWS account with permissions for SageMaker, S3, Lambda
- AWS CLI configured locally

### Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/DCode-v05/AWS-Machine-Learning-Foundations.git
   cd AWS-Machine-Learning-Foundations
   ```
2. **(Optional) Create a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Configure AWS CLI:**
   ```bash
   aws configure
   ```

## Usage
- Launch Jupyter Notebook:
  ```bash
  jupyter notebook
  ```
- Open any notebook (e.g., `3_2-machinelearning.ipynb`) and follow the instructions.
- For web demos, open the HTML files in `Lab 6/` in your browser.
- Ensure AWS credentials are set for any SageMaker or AWS service interaction.

## Folder Structure
```
AWS-Machine-Learning-Foundations/
├── 05-facedetection.ipynb           # Face detection (Computer Vision)
├── 3_2-machinelearning.ipynb        # ML fundamentals/data prep
├── 3_3-machinelearning.ipynb        # Categorical encoding lab
├── 3_4-machinelearning.ipynb        # Data split & XGBoost training
├── 3_5-machinelearning.ipynb        # Model deployment/batch transform
├── 3_6-machinelearning.ipynb        # Model evaluation/metrics
├── 3_7-machinelearning.ipynb        # Hyperparameter tuning
├── Flight_Delay-Student.ipynb       # Time series/forecasting lab
├── Vertebral Column.ipynb           # Vertebral column classification
└── Lab 6/
    ├── index.html                   # Chatbot web demo
    └── error.html                   # Error page for chatbot
```

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository
2. Create a new branch:
   ```bash
   git checkout -b feature/your-feature
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add your feature"
   ```
4. Push to your branch:
   ```bash
   git push origin feature/your-feature
   ```
5. Open a pull request describing your changes.

## Contact
For questions, suggestions, or feedback:
- **GitHub:** [DCode-v05](https://github.com/DCode-v05)
- **Email:** denistanb05@gmail.com

---
