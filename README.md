# News Reliability Detector

## Overview

Machine learning project that classifies news articles into reliable and unreliable categories. It leverages natural language processing (NLP) and machine learning algorithms for predictive analysis.

## Data Source

Data set for the project is available in Kaglle. You can find the dataset [here](https://www.kaggle.com/c/fake-news/data).

## Pipeline Diagram

![Pipeline Diagram](https://github.com/rnimisha/news-reliability-detector/tree/main/src/utils/pipeline.jpg?raw=true)

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
