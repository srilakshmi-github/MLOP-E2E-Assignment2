# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
data_size = 200
data = {
    'Age': np.random.randint(18, 70, size=data_size),
    'Salary': np.random.randint(30000, 150000, size=data_size),
    'Gender': np.random.choice(['Male', 'Female'], size=data_size),
    'Department': np.random.choice(['HR', 'Finance', 'Engineering', 'Marketing'], size=data_size)
}

# Create DataFrame
df = pd.DataFrame(data)

# Introduce missing values
df.loc[np.random.choice(df.index, size=10, replace=False), 'Salary'] = np.nan
df.loc[np.random.choice(df.index, size=5, replace=False), 'Age'] = np.nan

# Handle missing values
imputer = SimpleImputer(strategy='mean')
df[['Age', 'Salary']] = imputer.fit_transform(df[['Age', 'Salary']])

# One-hot encode categorical variables
df = pd.get_dummies(df, columns=['Gender', 'Department'], drop_first=True)

# Split data into features and target
X = df.drop(columns=['Age'])
y = df['Age']

# Split into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train[['Salary']] = scaler.fit_transform(X_train[['Salary']])
X_val[['Salary']] = scaler.transform(X_val[['Salary']])
X_test[['Salary']] = scaler.transform(X_test[['Salary']])