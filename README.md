# Credit Risk Prediction Model for Loan Default Classification
In the financial industry, risk management is crucial to maintaining profitability. One of the most prominent issues in this space is loan defaults. This project aims to develop a predictive model utilizing machine learning to classify whether a loan applicant will default based on several input factors. 

# Data Source
Dataset can be found on Kaggle: https://www.kaggle.com/datasets/nanditapore/credit-risk-analysis/data 

# Tools and Frameworks
- **Python:** pandas, NumPy
- **EDA:** pandas, SimpleImputer, matplotlib, seaborn
- **Machine Learning:** scikit-learn (Logistic Regression, Random Forest, XGBoost)

# Dataset Summary
- There are 32,581 values
- Class Imbalance: the dataset features a class imbalance which will be addressed in the modeling phase through techniques such as SMOTE to balance the classes. This is expected as in the context of loan defaults, there are typically not that many defaults compared to no defaults.
### Column Descriptions:
- ID: Unique identifier for each loan applicant.
- Age: Age of the loan applicant.
- Income: Income of the loan applicant.
- Home: Home ownership status (Own, Mortgage, Rent).
- Emp_Length: Employment length in years.
- Intent: Purpose of the loan (e.g., education, home improvement).
- Amount: Loan amount applied for.
- Rate: Interest rate on the loan.
- Status: Loan approval status (Fully Paid, Charged Off, Current).
- Percent_Income: Loan amount as a percentage of income.
- Default: Whether the applicant has defaulted on a loan previously (Yes, No).
- Cred_Length: Length of the applicant's credit history.

### Target Class Values
```python
df.Status.value_counts()
```
![image](https://github.com/user-attachments/assets/55946325-468e-44f6-8488-f2d54e1f025a)


### Dataset Info
![image](https://github.com/user-attachments/assets/72b97c6a-4e80-4e2b-bb13-20ed7086e412)

### Dataset Preview
![image](https://github.com/user-attachments/assets/fb255df1-0eaa-4e30-9e10-be79d020bb0a)

# Dataset Cleaning & Preprocessing
### Transforming Categorical Variables
```python
# We use pd.get_dummies function to transform our categorical columns using dummy variables

df_encoded = pd.get_dummies(df, columns=["Home", "Intent"], drop_first=True)
df_encoded['Default'] = [1 if i == "Y" else 0 for i in df['Default']]
```
![image](https://github.com/user-attachments/assets/b0f386ae-5326-47a0-9301-fe62d8506c8a)

### Handle Missing and Duplicate Values
```python
# We use SimpleImputer to fill in the missing values with mean
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df_encoded), columns=df_encoded.columns)
```
```python
# There is no duplicates so no action needed here
df_imputed.duplicated().sum()
```

### Drop Id Column
```python
# Id column does not provide any useful information so we drop it
# df_cleaned will be our initial model dataset from this point
df_cleaned = df_imputed.drop(["Id"], axis=1)
```

# Visualizations
### Boxplot Distribution of Numerical Columns
- Most of the numerical have significant outliers, such as Income, Age, and Employment Length.
<details>
  <summary>View Code</summary>

```python
fig, axes = plt.subplots(3, 2, figsize=(12, 10))

sns.boxplot(data = df, x="Default", y="Income", ax=axes[0,0])
axes[0,0].set_title("Default by Income")
axes[0,0].ticklabel_format(style='plain', axis='y')

sns.boxplot(data = df, x="Default", y="Age", ax=axes[0,1])
axes[0,1].set_title("Default by Age")

sns.boxplot(data = df, x="Default", y="Emp_length", ax=axes[1,1])
axes[1,1].set_title("Default by Employment Length")

sns.boxplot(data = df, x="Default", y="Rate", ax=axes[1,0])
axes[1,0].set_title("Default by Loan Rate")

sns.boxplot(data = df, x="Default", y="Amount", ax=axes[2,1])
axes[2,1].set_title("Default by Loan Amount")

sns.boxplot(data = df, x="Default", y="Cred_length", ax=axes[2,0])
axes[2,0].set_title("Default by Credit Length")

plt.tight_layout()
```
</details>

![image](https://github.com/user-attachments/assets/b19dfd43-f823-4204-9331-0a8e9960e753)

### Categorical Columns
- When viewing the cateogorical variables by default counts, it is clear there is a class imbalance of signifcantly more No Default values.
<details>
  <summary>View Code</summary>

```python
# We are using df because it contains the un-transformed categorical data
fig, axes = plt.subplots(4,1, figsize=(10,15))
sns.countplot(x=df["Home"], hue=df['Default'], ax=axes[0])
axes[0].set_title("Defaults by Home Category Countplot")

sns.countplot(x=df["Intent"], hue=df['Default'], ax=axes[1])
axes[1].set_title("Defaults by Intent Category Countplot")

sns.countplot(x=df["Status"], hue=df['Default'], ax=axes[2])
axes[2].set_title("Defaults by Status Category Countplot")

sns.countplot(x=df["Default"], ax=axes[3])
axes[3].set_title("Total Default Countplot")

plt.tight_layout()
plt.show()
```
</details>

![image](https://github.com/user-attachments/assets/977831a6-7ca2-4648-8a6d-884d87286d84)

### Target Feature Distribution
- A closer look at the distribution of Default and No Default values
<details>
  <summary>View Code</summary>

```python
# Pie chart for Loan Default Status with a closer view
plt.figure(figsize=(6, 6))
df['Default'].value_counts().plot.pie(autopct='%1.1f%%', colors=['green', 'lightgreen'])

plt.title('Distribution of Default')
plt.ylabel('')  # To remove the 'Default' label on the y-axis
plt.show()
```
</details>

![image](https://github.com/user-attachments/assets/9c365a03-c0c5-46c0-abc0-0197d6f3689a)

# Modeling
The goal was to develop a classification model capable of predicting loan defaults by classifying an applicant as likely to default or not default based on our dataset's variables. My modeling approach was to train and evaluate several machine learning models best suited for this classification task and fine-tune the best model based on benchmark performances such as classification report. 

## Model Evaluation
Random Forest, Logistic Regression, Narive Bayes, and XGBoost were implemented and their benchmark performance without any optimization is shown below through a AUROC graph.

### AUROC Graph
![image](https://github.com/user-attachments/assets/63b6ea21-fd20-4d43-953b-4b4bb3162964)


## Model Improvement
XGBoost was the best performing model. Several techinques were utilized to improve the model:
- Adjusting class weights by tuning the "scale_pos_weight" parameter in XGBoost
- Resampling using SMOTE to address the class imbalance
- Standardization using StandardScaler() to standardize the data

### Cost Matrices of Model Before and After Optimization
Cost matricies is utilized to evaluate the model's performance beyond accuracy. We are trying to **minimize false negative prediction** as falsely predicting an applicant as not defaulting when they actually do end up defaulting is much more costly than the other way around. The costs are calculated using the "Amount" and "Rate" columns for a simple interpretation.
- False Negative: Costs = Amount * (1+Rate)% 
- False Positive: Costs = Amount * Rate%

#### Before
![image](https://github.com/user-attachments/assets/6a30bd5e-e12f-40fe-9805-0492330eb597)
**Total Costs:** $8,008,985
#### After
![image](https://github.com/user-attachments/assets/d329b294-d8f8-4b7c-b220-a6957aa39911)
**Total Costs:** $3,160,103

# Results
The XGBoost model after optimization achieved the most ideal result across a combination of multiple evaluation metrics.
### Classification Report: 
| Class        | Precision | Recall  | F1-Score | Support |
|--------------|-----------|---------|----------|---------|
| 0.0          | 0.96      | 0.83    | 0.89     | 5322    |
| 1.0          | 0.52      | 0.83    | 0.64     | 1195    |
| **Accuracy** |           |         | 0.83     | 6517    |
| **Macro Avg**| 0.74      | 0.83    | 0.76     | 6517    |
| **Weighted Avg**| 0.87   | 0.83    | 0.84     | 6517    |

# Future Work
For further improvements or research on this model, several techniques can be implemented.
- Exploring other machine learning models, such as neural networks to assess whether it can improve prediction accuracy
- Incorporate external financial indicator data, such as credit scores and other demographic/macroeconomic factors to further enhance predictive power.

Thank you for reading!
