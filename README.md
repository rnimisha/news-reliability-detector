# News Reliability Detector with PySpark

## Overview

Machine learning project that classifies news articles into reliable and unreliable categories.

## Data Source

Data set for the project is available in Kaglle. You can find the dataset [here](https://www.kaggle.com/c/fake-news/data).

## Pipeline Diagram

![Pipeline Diagram](https://raw.githubusercontent.com/rnimisha/news-reliability-detector/main/src/utils/img/flowchart.jpeg)

## Model Evaluation

![Confusion Matrix](https://raw.githubusercontent.com/rnimisha/news-reliability-detector/main/src/utils/img/confusionmatrix.png)

1. Precision

- Out of all prediction as FAKE(1), Random Forest Model has highest precision
- Out of all prediction as NOT FAKE(0), Random Forest Model has highest precision

2. Recall

- Out of all actual FAKE(1), Logistic Regression Model is more accurate
- Out of all acutal NOT FAKE(0), Logistic Regression Model is more accurate

## Model Selection Insights

#### 1. Logistic Regression

- Strengths:

  - Balanced precision and recall for both classes.
  - Robust performance in predicting both fake and not-fake news.

#### 2. Random Forest

- Strengths:
  - Higher precision for Class 1, making it a good choice if minimizing false positives is crucial.
  - Moderate recall for Class 1.
- Key Point:

  - Improved performance when false positives are main concern.

#### 3. Naive Bayes

- Strengths:

  - Balanced performance with competitive precision and recall.

If minimizing false positives is important, Random Forest can be preferred.

If a balance between false positives and false negatives is essential, Logistic Regression and Naive Bayes are considerable options.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/rnimisha/news-reliability-detector

   cd news-reliability-detector
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset from Kaggle and place it in the `data/raw/` directory.

## Project Structure

The project follows a structured organization:

- `src/`: Reusable code.
- `data/`: Data directory for storing datasets.
- `models/`: Directory for saving trained models.
- `notebooks/`: For interactive eda, training and evaluation.
