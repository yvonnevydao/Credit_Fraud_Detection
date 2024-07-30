# Credit Card Fraud Detection

A machine learning project to detect fraudulent credit card transactions using both supervised and unsupervised learning techniques.

## Table of Contents

- [Introduction](#introduction)
- [Data Understanding](#data-understanding)
- [Data Preparation](#data-preparation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Modeling](#modeling)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)


## Introduction
Credit card fraud detection is crucial for financial institutions to minimize losses and protect customers. 

## Data Understanding

#### Data Collection (Source)
The dataset contains transactions made by credit cards in September 2013 by European cardholders, and also available in Kaggle: [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud). 
- The features include transaction time, amount, and anonymized variables.

#### Objective
This project explores various machine learning models to identify fraudulent transactions in a dataset of credit card transactions. 
- We **compare the performance** of **supervised** and **unsupervised learning techniques** to determine the most effective approach.

#### Steps:
- Load the dataset
- Data Types
- Class Distribution

![class_distribution_pie](https://github.com/user-attachments/assets/031335da-ac64-4231-9b78-a081832d99a8)

It contains 284,807 transactions, with 492 cases of fraud.
- As we can see, the dataset is **very imbalanced** with is about **0.17% Fraud** transaction.
  
=> Use this dataset for predictive models might cause a lot of errors because the models will tend to overfit since it gonna assume the most transactions are Non-Fraud

## Installation
```python
df = pd.read_csv('creditcard.csv')
display(df.head())
```
