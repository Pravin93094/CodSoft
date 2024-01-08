
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load your dataset (replace 'sales_data.csv' with your file path)
data = pd.read_csv(r"C:\Users\Pravin Andhale\Documents\advertising.csv")  # Update with your file path

# Select features and target variable
features = ['TV', 'Radio', 'Newspaper']  # Adjust feature columns based on your dataset
target = 'Sales'  # Replace with your target variable

X = data[features]
y = data[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test data
predictions = model.predict(X_test)

# Evaluate the model
print('Mean Squared Error:', mean_squared_error(y_test, predictions))
print('R-squared:', r2_score(y_test, predictions))
