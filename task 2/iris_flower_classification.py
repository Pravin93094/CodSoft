
# Import necessary libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Load the Iris dataset
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
target = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

# Scale the features (optional but can improve some models)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the Support Vector Classifier (SVC)
model = SVC(kernel='rbf', C=1, gamma='scale', random_state=42)

# Train the model on the training data
model.fit(X_train_scaled, y_train)

# Make predictions on the test data
predictions = model.predict(X_test_scaled)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions, target_names=iris.target_names)

# Print the evaluation metrics
print("Accuracy:", accuracy)
print("Classification Report:\n", report)
