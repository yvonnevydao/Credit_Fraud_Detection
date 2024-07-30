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
- The features include transaction `Time`, `Amount`, and `anonymized variables`.

#### Objective
This project explores various machine learning models to identify fraudulent transactions in a dataset of credit card transactions. 
- We **compare the performance** of **supervised** and **unsupervised learning techniques** to determine the most effective approach.

#### Steps:
- Load the dataset
- Data Types
- Class Distribution

![class_distribution_pie](https://github.com/user-attachments/assets/031335da-ac64-4231-9b78-a081832d99a8)

It contains `284,807 transactions`, with `492` cases of `fraud`.

⇒ As we can see, the dataset is **very imbalanced** with is about **0.17% Fraud** transaction.
  
⇒ Use this dataset for predictive models might cause a lot of errors because the models will tend to overfit since it gonna assume the most transactions are Non-Fraud

## Data Preparation
1. [Data Cleaning](#data-cleaning)
2. [Data Transformation](#data-transformation)
3. [Feature Selection](#feature-selection)

### 1. Data Cleaning 
- Check for `Null` value
- Handling Duplicates

⇒ There is no `Null` value

⇒ There are `1081 duplicated rows` in the dataset. Duplicate rows can skew the analysis and the results of the model trainning, as they can introduce bias and potentially lead to overfitting. 

⇒ Removing duplicates is generally a good practice in data preprocessing, especially in a sensitive task like fraud detection where accuracy is critical. So, I am gonna **remove** that.

⇒ The Class Distribution after remove Duplicates rows: `283,253 Non-fraud`, 
and `473 fraud`.

### 2. Data Transformation
- Variable Scaling: `Amount` and `Time`
- Correlation Matrix (Heatmap)

#### Before Scaling

![amount_time_distributionplot](https://github.com/user-attachments/assets/91167223-b818-473d-9e2f-3aa97e236c86)

`Amount` is not normally distributed ⇒ I've used `Log Transformation` as applying a  logarithmic transformation can help in reducing the skewness of the distribution. 

![scaled_Amount_distribution](https://github.com/user-attachments/assets/9aebb29b-516b-4027-ad78-8c747fe67345)

⇒ Im choosing  `StandardScaler` `Time` as `Time` is not really skewed.
![scaled_Time_distribution](https://github.com/user-attachments/assets/742d0651-10b4-4a5e-be72-c882927c54e5)

#### Compare Boxplots of Before and After Scaling
**Before:**
![amount_time_boxplot](https://github.com/user-attachments/assets/ae744ee6-0b0f-4ac4-9772-8fe1c5481aca)

**After:**
![boxplot_scaled_TimeAmount](https://github.com/user-attachments/assets/97239e9c-a93e-46a9-abcc-743732ba728b)

⇒ As we can see, there are still some **outliers** in `Amount_scaled` after scaling, but I decide not to dropping outliers based on the context of the analysis.

⇒ The outliers in the `Non-Fraud` class, so it kinda make sense and it might be **True Outliers** as the outliers are true reflections of the underlying population (e.g., rare but legitimate high-value transactions) ⇒ We might not want to remove the, as they provide valuable information.

#### Compare Correlation Matrix (Before and After `Scaling`)
**Correlation Matrix (Before Scaling)**
![correlationMatrix_before](https://github.com/user-attachments/assets/78d8a7a6-b975-4157-8bd7-c76d5cfb4796)

**Correlation Matrix (After Scaling)**
![correlationMatrix_after](https://github.com/user-attachments/assets/ce253e32-a0e3-45e6-8c6e-aac46d718373)

⇒ There is no Strong Correlation (larger than 0.70 and smaller than -0.70)

### 3. Feature Selection
- Define Predictors `X` and Target `y`
- Resampling: `None`, `Under-sampling`, `Combine`

