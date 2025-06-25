# CodeClauseInternship-Movie-Genre-Prediction

A beginner-friendly NLP + Machine Learning project for predicting the genre of a movie based on its plot description. This project uses text preprocessing, TF-IDF vectorisation, and classification models like Naive Bayes, Logistic Regression, and Random Forest. Built as part of the CodeClause Data Science Internship.

## What This Project Does

- Predicts the movie genre based on its plot summary.
- Uses Natural Language Processing (NLP) to clean and process text.
- Converts movie plot descriptions into numerical features using TF-IDF.
- Trains multiple classifiers to predict genres:
  - **Naive Bayes**
  - **Logistic Regression**
  - **Random Forest**
- Evaluates models using accuracy, F1-score, and a confusion matrix.
- Bonus: Discusses the possibility of multi-label classification.

## Technologies Used

- Programming Language: Python  
- Libraries & Tools:
  - [NLTK](https://www.nltk.org/) – for natural language processing (tokenization, lemmatization, stopwords)
  - [Scikit-learn](https://scikit-learn.org/) – for machine learning models, TF-IDF vectorizer, evaluation metrics
  - [Matplotlib](https://matplotlib.org/) & [Seaborn](https://seaborn.pydata.org/) – for visualization

## Project Structure

### 1. Import Libraries & Load Dataset
The dataset used is **IMDb Movie Data** which includes columns like `Title`, `Genre`, `Description`, etc.

### 2. Preprocess the Text
- Lowercase conversion
- Tokenization
- Removing punctuation and numbers
- Stopword removal
- Lemmatization (using NLTK's WordNet)

### 3. Vectorize the Text (TF-IDF)
TF-IDF is used to convert plot descriptions into numeric format that can be used for training.

### 4. Encode Target Labels
Genres are encoded using `LabelEncoder`. Only the **first genre** is used for simplification.

### 5. Train Classifiers
- **Naive Bayes**: Simple and fast
- **Logistic Regression**: Better balance between precision and recall
- **Random Forest**: Captures nonlinear patterns better

### 6. Evaluate the Models
- Accuracy Score
- Classification Report (Precision, Recall, F1-score)
- Confusion Matrix for Logistic Regression

### 7. Bonus Discussion
- Why multi-label classification might be a better approach
- Limitations due to class imbalance

## Sample Results

| Model              | Accuracy |
|-------------------|----------|
| Naive Bayes        | ~30.5%   |
| Logistic Regression| ~39.0%   |
| Random Forest      | ~39.5%   |

Genres like **Action** and **Drama** performed better, while **Thriller**, **Mystery**, and **Animation** were difficult due to fewer samples.

## How to Run This Project

**Clone the Repository**:
```bash
git clone https://github.com/Gowry11/CodeClauseInternship-Movie-Genre-Prediction-2-.git
```
**Run the Notebook**:
- Open MovieGenrePrediction.ipynb in Jupyter Notebook or any Python IDE.
- Run all cells step-by-step.

 ## What You’ll Learn
- Basics of NLP text cleaning and feature extraction
- Applying TF-IDF for vectorizing text
- Training and evaluating ML models for classification
- Understanding model performance using precision, recall, F1-score
- Challenges in real-world datasets (imbalance, multi-label)

## Developed By  
Gowry P P
