# Income Prediction Analysis Project

## Overview
This project analyzes the "adult.csv" dataset to predict whether an individual's income exceeds $50,000 per year based on demographic and employment factors. The analysis includes extensive exploratory data analysis (EDA), implementation of multiple preprocessing pipelines, and comparison of Logistic Regression and SVM classifiers.

## Dataset Features
- Age
- Workclass
- Education
- Occupation
- Race
- Sex
- Income (Target variable: >50K or ≤50K)
- Capital gain/loss
- Hours per week
- Native country

## Key Findings from EDA

### Correlations
- Highest correlation (0.15) between education_num and hours_per_week
- Positive correlation (0.12) between capital gain and education_num
- Minimal correlation (-0.078) between age and fnlwgt

### Demographic Insights
- Age: Individuals over 40 tend to earn more than 50K
- Gender: Significant income disparity between males and females
- Race: Whites form the largest group with diverse income distribution
- Location: US residents show highest income diversity

## Preprocessing Pipelines

### Imputation Methods
1. Mean Imputation
2. Mode Imputation
3. Linear Regression Imputation

### Scaling Methods
1. Log Scaling: Addresses skewness and outliers
2. Robust Scaling: Standardizes data using median and IQR

## Model Performance

### Logistic Regression Results
| Pipeline | Accuracy | Precision | Recall | F1-score |
|----------|----------|-----------|--------|-----------|
| P1 (LR+Log+L1) | 69.88% | 26.33% | 10.94% | 15.46% |
| P2 (LR+Robust) | 70.63% | 24.68% | 8.54% | 24.68% |
| P3 (Mean+Log+L2) | 69.24% | 23.97% | 10.11% | 14.22% |
| P4 (Mean+Robust) | 69.74% | 28.39% | 9.31% | 14.02% |

### SVM Results
| Pipeline | Accuracy | Precision | Recall | F1-score |
|----------|----------|-----------|--------|-----------|
| P1 (LR+Log+L1) | 78.81% | 64.98% | 34.07% | 44.70% |
| P2 (LR+Robust) | 76.23% | 52.06% | 64.16% | 57.48% |
| P3 (Mean+Log+L2) | 76.95% | 74.36% | 15.49% | 25.64% |
| P4 (Mean+Robust) | 75.27% | 49.62% | 59.17% | 53.98% |

## Best Performing Models
- Logistic Regression: P2 (Linear Regression Imputation + Robust Scaling)
- SVM: P2 (Linear Regression Imputation + Robust Scaling)

## Use Case Recommendations

### Logistic Regression Pipelines
- P1: Suitable for cases requiring high positive capture rate
- P2: Ideal for precision-focused applications
- P3: Good baseline model for initial analysis
- P4: Balanced performance for general applications

### SVM Pipelines
- P1: Best for screening applications with tolerance for false positives
- P2: Optimal for balanced accuracy and recall requirements
- P3: Suitable for precision-critical applications
- P4: Good for general-purpose predictions requiring balance

## Requirements
- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Seaborn

## Usage
1. Load the adult.csv dataset
2. Run preprocessing pipelines
3. Train models
4. Evaluate results
5. Choose appropriate pipeline based on use case requirements
