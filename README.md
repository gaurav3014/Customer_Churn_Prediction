
# Customer_Churn_Prediction

## Problem Definition
The Customer Churn table contains information on all 7,043 customers from a Telecommunications company in California in Q2 2022.

Each record represents one customer and contains details about their demographics, location, tenure, subscription services, status for the quarter (joined, stayed, or churned), and more!

The Zip Code Population table contains complementary information on the estimated populations for the California zip codes in the Customer Churn table.

We need to predict whether the customer will churn, stay or join the company based on the parameters of the dataset.
<br><br>

## 📓 Overview

| Machine Learning Models Applied | Accuracy |
| ------------------------------- | -------- |
| Random Forest                   | 78.11%   |
| Logistic Regression             | 78.28%   |
| Naive Bayes Gaussian            | 36.77%   |
| Decision Tree                   | 77.29%   |
| XGB_Classifier                  | 80.86%   |

<br>

## 👉 Application

The ability to predict churn before it happens allows businesses to take proactive actions to keep existing customers from churning. This could look like:
```
  Customer success teams reaching out to those high-risk customers to provide support or to gauge 
  what needs may not be being met.
```

The advantage of calculating a company's churn rate is that it provides clarity on how well the business is retaining customers, which is a reflection on the quality of the service the business is providing, as well as its usefulness.

<br>

## Importing Necessary Files

```python
import numpy as np
import pandas as pd
df = pd.read_csv('telecom_customer_churn.csv')
df.head(5)
```

## Data Overview

The first few rows of the dataset:
```
Customer ID | Gender | Age | Married | Number of Dependents | City         | Zip Code | Latitude  | Longitude  | Number of Referrals | ... | Payment Method | Monthly Charge | Total Charges | Total Refunds | Total Extra Data Charges | Total Long Distance Charges | Total Revenue | Customer Status | Churn Category | Churn Reason
------------|--------|-----|---------|----------------------|--------------|----------|-----------|------------|----------------------|-----|----------------|----------------|---------------|---------------|--------------------------|-----------------------------|---------------|-----------------|----------------|--------------
0002-ORFBO  | Female | 37  | Yes     | 0                    | Frazier Park | 93225    | 34.827662 | -118.999073 | 2                   | ... | Credit Card    | 65.6           | 593.30        | 0.00          | 0                        | 381.51                      | 974.81        | Stayed          | NaN            | NaN
0003-MKNFE  | Male   | 46  | No      | 0                    | Glendale     | 91206    | 34.162515 | -118.203869 | 0                   | ... | Credit Card    | -4.0           | 542.40        | 38.33         | 10                       | 96.21                       | 610.28        | Stayed          | NaN            | NaN
0004-TLHLJ  | Male   | 50  | No      | 0                    | Costa Mesa   | 92627    | 33.645672 | -117.922613 | 0                   | ... | Bank Withdrawal| 73.9           | 280.85        | 0.00          | 0                        | 134.60                      | 415.45        | Churned         | Competitor     | Competitor had better devices
0011-IGKFF  | Male   | 78  | Yes     | 0                    | Martinez     | 94553    | 38.014457 | -122.115432 | 1                   | ... | Bank Withdrawal| 98.0           | 1237.85       | 0.00          | 0                        | 361.66                      | 1599.51       | Churned         | Dissatisfaction| Product dissatisfaction
0013-EXCHZ  | Female | 75  | Yes     | 0                    | Camarillo    | 93010    | 34.227846 | -119.079903 | 3                   | ... | Credit Card    | 83.9           | 267.40        | 0.00          | 0                        | 22.14                       | 289.54        | Churned         | Dissatisfaction| Network reliability
```

## Exploratory Data Analysis

Overview of all columns in the dataset:
```python
df.columns
```

Creating a copy of the dataset:
```python
df1 = df.copy()
df1.head(7)
```

## Data Preprocessing

Dropping unwanted columns from the dataset:
```python
df1.drop(['Customer ID', 'Total Refunds', 'Zip Code', 'Latitude', 'Longitude', 'Churn Category', 'Churn Reason'], axis='columns', inplace=True)
df1.shape
```

Checking the data types of each column:
```python
df1.dtypes
```

Checking the number of unique values in each column:
```python
features = df1.columns
for feature in features:
     print(f'{feature}--->{df[feature].nunique()}')
```

Getting the percentage of Null Values in each column:
```python
df1.isnull().sum() / df1.shape[0]
```

Cleaning Function for the Dataset:
```python
def clean_dataset(df):
    assert isinstance(df, pd.DataFrame)
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

df1 = df1.interpolate()
df1 = df1.dropna()
```

Checking unique values of columns having datatype: 'object':
```python
def unique_values_names(df):
    for column in df:
        if df[column].dtype == 'object':
            print(f'{column}: {df[column].unique()}')
unique_values_names(df1)
```

## Data Visualization

### Visualizing Column 'Age' in the dataset
```python
import plotly.express as px
fig = px.histogram(df1, x = 'Age')
fig.show()
```

### Checking the stats in number_columns of the copied dataset
```python
df1.hist(figsize=(15, 15), xrot=30)
```

### Visualizing the number of customers who churned, stayed or joined in the company with a bar plot
```python
import matplotlib.pyplot as plt

Customer_Stayed = df1[df1['Customer_Status'] == 'Stayed'].Age
Customer_Churned = df1[df1['Customer_Status'] == 'Churned'].Age
Customer_Joined = df1[df1['Customer_Status'] == 'Joined'].Age

plt.xlabel('Age')
plt.ylabel('Customers Numbers')
plt.hist([Customer_Stayed, Customer_Churned, Customer_Joined], color=['black', 'red', 'blue'], label=['Stayed', 'Churned', 'Joined'])
plt.title('Customers Behavior ', fontweight="bold")
plt.legend()
plt.show()
```

### Defining Correlation between the columns in the dataset
```python
import seaborn as sns

data = df1.corr()
plt.figure(figsize=(20, 10))
sns.heatmap(data, annot=True)
plt.show()
```

### Analyzing Outlier in the dataset with respect to customer status
```python
fig, ax = plt.subplots(4, 3, figsize=(15, 15))
for i, subplot in zip(number_columns, ax.flatten()):
    sns.boxplot(x='Customer_Status', y=i, data=df1, ax=subplot)

plt.tight_layout()
plt.show()
```

### Density Heatmap
```python
fig = px.density_heatmap(df1, x='Age', y='Total Charges')
fig.show()
```

### Cross Tabulations
```python
pd.crosstab(df['Customer_Status'], df['Married']).plot(kind='bar')
pd.crosstab(df['Customer_Status'], df['Gender']).plot(kind='bar')
plt.show()
```

## Data Modeling

### Replacing the Gender column in the dataset with Label Encoding
```python
df1.replace({"Gender": {'Female': 0, 'Male': 1}}, inplace=True)
```

### Replacing the columns with 'yes' and 'no' output by Label Encoding
```python
yes_and_no = ['Paperless Billing', 'Unlimited Data', 'Streaming Movies', 'Streaming Music', 'Streaming TV', 'Premium Tech Support', 'Device Protection Plan', 'Online Backup', 'Online Security', 'Multiple Lines', 'Married']
for column in yes_and_no:
    df1.replace({column: {'No': 0, 'Yes': 1}}, inplace=True)
```

