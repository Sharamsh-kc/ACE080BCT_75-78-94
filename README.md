# ACE080BCT_75-78-94
## GROUP: WILL OF D(DATA)

- **Roll Numbers: ACE080BCT075, ACE080BCT078,ACE080BCT094**

# Advanced Logistic Regression for Loan Approval Prediction

## Project Overview
This project implements an **advanced Logistic Regression model from scratch** to predict **loan approval status** based on financial attributes.This is an updated repository. Instead of using `sklearn.linear_model.LogisticRegression`, we have implemented a **custom Logistic Regression algorithm** using **NumPy**.

## Dataset Details
The dataset consists of 500 samples with the following attributes:
- **Income**: Monthly salary in dollars
- **Credit Score**: Range from 300 to 850
- **Loan Amount**: Amount requested for the loan
- **Debt-to-Income Ratio**: Percentage of income spent on debts
- **Approval Status**: Target variable (1 = Approved, 0 = Not Approved)

## Code Explanation
### **1. Data Preprocessing**
- **Feature Scaling**: Standardized the numerical features using `StandardScaler()`.
- **Polynomial Features**: Created quadratic features using `PolynomialFeatures(degree=2, include_bias=False)`. This allows the model to capture non-linear relationships.

### **2. Custom Logistic Regression Implementation**
- We implemented **Logistic Regression from scratch** without using `sklearn.linear_model.LogisticRegression`.
- The **sigmoid function** is used for prediction.
- **Gradient Descent Optimization** is applied to update weights and bias.
- The training loop iteratively updates parameters to minimize the loss function.

### **3. Model Training & Evaluation**
- **Accuracy, Confusion Matrix, and Classification Report**: Displayed standard classification metrics.
- **ROC-AUC Score**: Evaluated the model’s ability to distinguish between approved and non-approved loans.

### **4. Model Saving**
- Saved the trained model using `joblib.dump()` so it can be reused without retraining.

## Installation & Usage
### **Requirements**
Ensure you have Python installed along with the required libraries:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn joblib
```

### **Running the Model**
1. Clone the repository:
```bash
git clone https://github.com/your-github-repo.git
cd your-github-repo
```
2. Run the script:
```bash
python logistic_regression_advanced.py
```
3. The model will output accuracy, confusion matrix, and classification report.

### **Using the Saved Model**
If you want to use the trained model for predictions without retraining:
```python
import joblib
import numpy as np

# Load the model
model = joblib.load("logistic_regression_custom.pkl")

# Example input (scaled appropriately before usage)
new_data = np.array([[8000, 700, 15000, 0.3]])  # Income, Credit Score, Loan Amount, Debt-to-Income Ratio
prediction = model.predict(new_data)
print("Loan Approval Status:", "Approved" if prediction[0] == 1 else "Not Approved")
```

## Repository Structure
```
├── logistic_regression_advanced.py  # Main script for training and evaluation
├── logistic_regression_loan_data.csv  # Dataset
├── logistic_regression_custom.pkl  # Saved trained model
├── README.md  # Project documentation
```

# References

## Datasets & Data Processing
1. Scikit-Learn: `StandardScaler()` documentation - [https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
2. Polynomial Features explanation - [https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html)

## Model Training & Evaluation
3. Logistic Regression in Scikit-Learn - [https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
4. GridSearchCV for hyperparameter tuning - [https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)

## Performance Metrics & Visualization
5. ROC Curve explanation - [https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html](https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html)
6. Confusion Matrix and Classification Report - [https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)

## Model Deployment & Saving
7. Using `joblib` for model persistence - [https://joblib.readthedocs.io/en/latest/](https://joblib.readthedocs.io/en/latest/)


## License
This project is for educational purposes only.

