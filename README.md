# Student Performance Classification

## Overview
This project uses a Decision Tree Classifier to classify students' final mathematics grades into four performance categories: Very Bad, Bad, Average, and Good.

The model is trained on the Student Performance dataset and uses demographic, social, and academic features to predict students' final grade category.

## Project Details
- **Machine Learning Task:** Classification
- **Algorithm:** Decision Tree Classifier
- **Maximum Tree Depth:** 5
- **Dataset:** Student Performance (Mathematics)
- **Train-Test Split:** 70% training, 30% testing

## Grade Categories
The final grade (`G3`) is transformed into four categories:
- Very Bad
- Bad
- Average
- Good

## Features
The model uses 32 features, including:
- Student demographics
- Family background
- Study habits
- School-related information
- Previous grades (`G1` and `G2`)
- Absences

## Evaluation
The model is evaluated using:
- Accuracy Score
- Classification Report
- Confusion Matrix

## Libraries
- Python
- Pandas
- NumPy
- Scikit-learn

## Purpose
This project was developed as a practical machine learning exercise to improve my understanding of data preprocessing, feature selection, classification models, and model evaluation.
