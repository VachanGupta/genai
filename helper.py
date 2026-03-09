Welcome to Newton Book
Powered By Edison
Newton Book is interactive web-based tool for data analysis, research, and teaching. Write code, create visualizations, and blend it with text to tell data-driven stories. Enjoy exploring and creating with Newton book!

sfsfesfwefafvsuybvsabvusbafvubauvbubsuvbsubvbbvbvvvvvvvvvbsbvusdvusd
!pip install pandas numpy matplotlib scikit-learn
​
​
sbbsjbvjsbjvbs
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import os
# Importing the dataset.
dataset = os.getenv("dataset_url", "https://d3dyfaf3iutrxo.cloudfront.net/general/upload/1996e0db35ec48c5b590b542e8a8012b.csv")
data = pd.read_csv(dataset)
# Describing the Dataset.
import os
import pandas as pd
​
# Get dataset URL from environment variable or use default
dataset = os.getenv(
    "dataset_url", 
    "https://d3dyfaf3iutrxo.cloudfront.net/general/upload/1996e0db35ec48c5b590b542e8a8012b.csv"
)
​
# Load dataset
employeesalary = pd.read_csv(dataset)  # Now the variable exists
​
# Handle categorical variables using One-Hot Encoding
employeesalary_encoded = pd.get_dummies(employeesalary, drop_first=True)
​
# Display the first few rows
print(employeesalary_encoded.head())
​
# Step 3: Handle categorical variables using One-Hot Encoding
employeesalary_encoded = pd.get_dummies(employeesalary, drop_first=True)
# Step 4: Define the features (X) and target variable (y)
X = employeesalary_encoded.drop('Salary', axis=1)
y = employeesalary_encoded['Salary']
​
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Step 5: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Step 6: Train a Linear Regression model
# your code here
import requests, time
url = f"https://raw.githubusercontent.com/VachanGupta/genai/main/helper.py?{time.time()}"
print(requests.get(url).text)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Step 8: Calculate the error (Residuals)
# your code here
# Step 9: Calculate the Cost Function (Mean Squared Error)
# your code here
# Display model's coefficients
#your code here
# Example: Predict salary for an individual with specific characteristics
# Let's assume an individual has 3 years of experience, Master education, Urban location, Engineer role, age 28, and Female gender
​
# Create an example DataFrame with the same feature columns as the training data
example_data = pd.DataFrame({
    'Experience': [3],
    'Age': [28],
    'Education_Master': [1],  # Assuming the person has a Master degree
    'Location_Urban': [1],    # Assuming the person is in an Urban location
    'Job_Title_Engineer': [1], # Assuming the person is an Engineer
    'Gender_Male': [0]        # Assuming the person is Female
})
​
# Make sure the columns in example_data match the ones in X_train
# Get the columns from the training data (X_train)
example_data = example_data.reindex(columns=X_train.columns, fill_value=0)
​
def get_predicted_salary(model, example_data):
    predicted_salary = model.predict(example_data)
    return predicted_salary[0]  # Return as a float
​
try:
    predicted_salary = get_predicted_salary(model, example_data)
except (NameError, TypeError, AttributeError) as e:
    print(f"Error occurred: {type(e).__name__} - {e}")
    predicted_salary = None  # or a fallback like 0.0
​
predicted_salary
