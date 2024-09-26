# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Set a random seed for reproducibility of the results
np.random.seed(42)

# Generate synthetic dataset with random values for Age, Salary, Gender, and Department
data_size = 200
data = {
    'Age': np.random.randint(18, 70, size=data_size),  # Age ranges between 18 and 70
    'Salary': np.random.randint(30000, 150000, size=data_size),  # Salary between 30K and 150K
    'Gender': np.random.choice(['Male', 'Female'], size=data_size),  # Randomly assign Gender
    'Department': np.random.choice(['HR', 'Finance', 'Engineering', 'Marketing'], size=data_size)  # Assign random departments
}

# Create a pandas DataFrame from the synthetic data
df = pd.DataFrame(data)

# Introduce missing values randomly in 'Salary' (10 missing) and 'Age' (5 missing)
df.loc[np.random.choice(df.index, size=10, replace=False), 'Salary'] = np.nan
df.loc[np.random.choice(df.index, size=5, replace=False), 'Age'] = np.nan

# Impute missing values in 'Age' and 'Salary' using the mean value for each column
imputer = SimpleImputer(strategy='mean')
df[['Age', 'Salary']] = imputer.fit_transform(df[['Age', 'Salary']])

# Apply one-hot encoding to the categorical columns 'Gender' and 'Department', dropping the first category to avoid multicollinearity
df = pd.get_dummies(df, columns=['Gender', 'Department'], drop_first=True)

# Split the dataset into features (X) and target variable (y)
X = df.drop(columns=['Age'])  # Features are all columns except 'Age'
y = df['Age']  # Target variable is 'Age'

# Perform a 60/40 split to create training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)

# Further split the remaining 40% into validation (20%) and test (20%) sets
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Scale the 'Salary' feature in training, validation, and test sets using standardization (z-score normalization)
scaler = StandardScaler()
X_train[['Salary']] = scaler.fit_transform(X_train[['Salary']])  # Fit and transform training data
X_val[['Salary']] = scaler.transform(X_val[['Salary']])  # Transform validation data
X_test[['Salary']] = scaler.transform(X_test[['Salary']])  # Transform test data
