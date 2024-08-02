# Credit Card Fraud Detection

A machine learning project to detect fraudulent credit card transactions using both supervised and unsupervised learning techniques.

## Table of Contents

- [Introduction](#introduction)
- [Data Understanding](#data-understanding)
- [Data Preparation](#data-preparation)
- [Modeling](#modeling)
- [Project Structure](#project-structure)
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
  
Number `column` of the dataset: `284,807`
Number `row` of the dataset: `31`

![class_distribution_pie](https://github.com/user-attachments/assets/031335da-ac64-4231-9b78-a081832d99a8)

It contains `284,807 transactions`, with `492` cases of `fraud`.

⇒ As we can see, the dataset is **very imbalanced** with is about **0.17% Fraud** transaction.
  
⇒ Use this dataset for predictive models might cause a lot of errors because the models will tend to overfit since it gonna assume the most transactions are Non-Fraud

## Data Preparation
[1. Data Cleaning](#data-cleaning)

[2. Data Transformation](#data-transformation)

[3. Feature Selection](#feature-selection)

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
- 3.1 Define Predictors `X` and Target `y`
- 3.2 Feature Selection on `X`
- 3.3 Split the data into training and test sets
- 3.4 Re-sampling on the training set: `None`, `Under-sampling`, `Combine`
- 3.5 Dimensionality Reduction and Clustering (Visualization)

#### 3.2 Feature Selection on `X` only
There are a couple options for feature selection when working with a `PCA dataset`:
- `Manual Feature Selection`: This involves using domain knowledge to select the most relevant principal components. ⇒ This approach is beneficial if we have a deep understanding of the domain and can interpret the components effectively.
- `Automated Feature Selection`: We can use Machine Learning techniques to automate feature selection process. ⇒ These techniques can help us identify the most important features based on statistical methods or model-based approaches without requiring extensive domain knowledge.

⇒ I am gonna `use Automated Feature Selection` as the features are almost anonymized

**Feature Selection Techniques:** Use one or a combination of `Filter`, `Wrapper`, and `Embedded methods`.

#### 3.3 Split the data into training and test sets
I use `random_state=seed` and `test_size=0.3` 

![class_distribution_Trainset](https://github.com/user-attachments/assets/2af3d97c-e25a-4da8-b0fe-57a6012e38a0)

Class Distribution (Training set): `198,290 Non-Fraud` and `318 Fraud`

![class_distribution_Testset](https://github.com/user-attachments/assets/10204a38-730b-428d-abfd-a6425c17431f)

Class Distribution (Test set): `84,963 Non-Fraud` and `155 Fraud`

#### 3.4.1 `Under-sampling`
There are many techniques for `Under-sampling`, and I am using `RandomUnderSampler` in this case.

![Undersampling](https://github.com/user-attachments/assets/d2fb75a6-4c0f-49ef-8320-e7800bd14030)

⇒ Under-sampling reduced the size of **Non-Fraud**, and the class distribution on the trainning set: `318 Fraud` and `318 Non-Fraud`

⇒ `Very Fast` to Train models

#### 3.4.2 Combine `Over-sampling` and `Under-sampling`
I've created a Pipeline for combine of Over-sampling and Under-sampling using: 
- `RandomOverSampler(sampling_strategy=0.01)`
- and `RandomUnderSampler(sampling_strategy=0.05)`

**NOTE:** If we are using Combine method without changing `sampling_strategy`, it will **increase** the `Fraud` transactions = `Non-Fraud` transactions. ⇒ 0.5 Fraud and 0.5 Non-Fraud

![class_distribution_COMBINE](https://github.com/user-attachments/assets/12127079-04a4-4210-8923-750832636f25)

⇒ Under-sampling reduced the size of **Non-Fraud** and Over-sampling increased the size of **Fraud**. 

⇒ The class distribution on the trainning set: `39,640 Non-Fraud` and `1,982 Fraud`

⇒ The Re-sampling dataset is not really Balanced, and it required a lot of time to trainning models

#### 3.5 Dimensionality Reduction and Clustering (Visualization)
Try `t-SNE`, `PCA`, and `SVD` on Training Predictors (`Under-sampling`):

- T-SNE took: 1.62 s
- PCA took: 0.0042 s
- TruncatedSVD took: 0.0036 s
  
![Visualizatio_Undersampling](https://github.com/user-attachments/assets/eb476383-ce2f-40ce-8e15-73839e6c189c)

Try `t-SNE`, `PCA`, and `SVD` on Training Predictors (`COMBINE`):

- T-SNE took: 94.65 s
- PCA took: 0.0058 s
- TruncatedSVD took: 0.0814 s
  
⇒ As we can see, T-SNE took more time to cluster, but it seem to be better at clustering??

![Visualizatio_COMBINE](https://github.com/user-attachments/assets/43dafab6-e3f1-42bf-b6b9-90f73ab7d219)


⇒ I don't even want to try `t-SNE`, `PCA`, and `SVD` on `Non-Resampling` dataset, as it took a lot of time to run. 

## Modeling

[1. Learning Curves (on each dataset)](#learning-curves)

[2. Choose Model](#choose-model)

[3. Feature Selection](#feature-selection)

### 1. Learning Curves (on each dataset)

Learning Curves for `LogisticRegression` for each dataset: `Non-Resampling`, `Under-Sampling`, and `Combine`

I am using `KFold(n_splits=5)` 

#### Compare Learning Curves for `Non-Resampling` and `Undersampling` dataset

**Non-Resampling Dataset:**

![learningCurve_NON_SAMPLING](https://github.com/user-attachments/assets/51378759-8592-4fb2-866a-08b89a648285)

⇒ Higher overall accuracy for both training and cross-validation scores.

⇒ Very small gap between training and cross-validation scores, indicating low bias and low variance.

⇒ More samples leading to stable and high performance.

**Undersampling (Resampling) Dataset:**

![learningCurve_UNDERSAMPLING](https://github.com/user-attachments/assets/7505950e-8261-46e9-ad55-5f0299cc82b1)

⇒ Lower cross-validation accuracy compared to the non-resampling dataset.

⇒ Larger gap between training and cross-validation scores, indicating higher variance and potential overfitting.

⇒ Stable but lower performance on unseen data with fewer samples.

**CONCLUSION:**

⇒ The learning curve for the non-resampling dataset is better. It shows higher accuracy, both in training and cross-validation, and a smaller gap between the two, indicating better generalization and less overfitting. The undersampling dataset's learning curve indicates more variance and lower generalization capability.
