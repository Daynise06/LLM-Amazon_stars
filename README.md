![](UTA-DataScience-Logo.png)

# LLM-Amazon_stars

* **One Sentence Summary** This repository holds an attempt to classify and predict product review sentiment and star ratings using the Amazon Product Reviews Dataset, leveraging both classical ML (TF-IDF + Logistic Regression) and LLM-based feature extraction.
https://www.kaggle.com/datasets/gzdekzlkaya/amazon-product-reviews-dataset/data

## Overview
The task, as defined by the Kaggle dataset, is to use Amazon customer review text and metadata to predict the product’s star rating (1–5) and overall sentiment. The approach in this repository formulates the problem as both:

* Multiclass classification (predicting the exact star rating)
* Binary classification (positive vs. negative sentiment)

We compare the performance of several models:
* TF-IDF + Linear Models
* SentenceTransformer embeddings + Logistic Regression
* Direct pretrained LLM sentiment classifiers

Our best binary model (TF-IDF + LinearSVC) achieved over 90% accuracy on a held-out test set, demonstrating the feasibility of this dataset for sentiment classification.

![](model_results.txt)

## Summary of Workdone

### Data

* Data:
  * Type: CSV of Amazon product reviews with fields such as:
    * review (text of the review)
    * star_rating (1–5 stars)
    * Additional optional fields: review summary, product category, helpfulness votes
  * Size: ~4,900 reviews after cleaning
  * Splits:
    * Multiclass: Train/Validation/Test = 80% / 20%
    * Binary: Same split after removing 3-star “neutral” reviews
 
#### Preprocessing / Clean up

* Renamed columns (overall → star_rating, reviewText → review)
* Dropped rows with missing or empty text
* Clipped ratings to 1–5 range
* Created binary sentiment label:
  * Positive: 4–5 stars
  * Negative: 1–2 stars
  * Neutral (3 stars) removed from binary dataset


#### Data Visualization

Key visualizations generated:
* Bar chart of star rating distribution
* Pie chart of binary sentiment split
* Histogram and violin plots of review lengths
* Top unigrams/bigrams overall and by sentiment
* Confusion matrices for binary and multiclass models
* ROC curve for binary classifier
* Word clouds for positive/negative reviews

******


### Problem Formulation

* Define:
  * Input: Review text (optionally with product metadata)
  * Output:
    * Multiclass: 1–5 star rating
    * Binary: Positive (4–5) or Negative (1–2)
  * Models:
    * TF-IDF + Logistic Regression (multiclass)
    * TF-IDF + LinearSVC (binary)
    * SentenceTransformer embeddings + Logistic Regression
    * Pretrained nlptown/bert-base-multilingual-uncased-sentiment pipeline
  * Metrics:
   * Accuracy
   * Macro F1-score
 
  
### Training

* Describe the training:
  * Environment: Google Colab (CPU/GPU)
  * Libraries: transformers, sentence-transformers, scikit-learn, pandas, numpy, matplotlib, seaborn
  * Duration:
    * Classical ML models: seconds
    * Embedding-based models: minutes
    * Pretrained LLM: minutes (depends on GPU availability)
  * Early stopping determined by monitoring validation accuracy/F1

 
### Performance Comparison

* Clearly define the key performance metric(s).
* Show/compare results in one table.
* Show one (or few) visualization(s) of results, for example ROC curves.

### Conclusions

* State any conclusions you can infer from your work. Example: LSTM work better than GRU.

### Future Work

* What would be the next thing that you would try.
* What are some other studies that can be done starting from here.

## How to reproduce results

* In this section, provide instructions at least one of the following:
   * Reproduce your results fully, including training.
   * Apply this package to other data. For example, how to use the model you trained.
   * Use this package to perform their own study.
* Also describe what resources to use for this package, if appropirate. For example, point them to Collab and TPUs.

### Overview of files in repository

* Describe the directory structure, if any.
* List all relavent files and describe their role in the package.
* An example:
  * utils.py: various functions that are used in cleaning and visualizing data.
  * preprocess.ipynb: Takes input data in CSV and writes out data frame after cleanup.
  * visualization.ipynb: Creates various visualizations of the data.
  * models.py: Contains functions that build the various models.
  * training-model-1.ipynb: Trains the first model and saves model during training.
  * training-model-2.ipynb: Trains the second model and saves model during training.
  * training-model-3.ipynb: Trains the third model and saves model during training.
  * performance.ipynb: loads multiple trained models and compares results.
  * inference.ipynb: loads a trained model and applies it to test data to create kaggle submission.

* Note that all of these notebooks should contain enough text for someone to understand what is happening.

### Software Setup
* List all of the required packages.
* If not standard, provide or point to instruction for installing the packages.
* Describe how to install your package.

### Data

* Point to where they can download the data.
* Lead them through preprocessing steps, if necessary.

### Training

* Describe how to train the model

#### Performance Evaluation

* Describe how to run the performance evaluation.


## Citations

* Provide any references.
