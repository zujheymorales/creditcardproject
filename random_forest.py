#Import Libraries 
import pandas as pd
import matplotlib.pyplot as plt
import csv
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Load Dataset
df = pd.read_csv('clean_dataset.csv')

# Drop non-numeric or irrelevant columns
cols_to_drop = ['Industry', 'Ethnicity', 'Citizen']
df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

#Split into Features and Target 
X = df.drop(columns=['Approved'])
y = df['Approved']

# Split into training and testing sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2
)

# Train Random Forest Classifier 
rf_model = RandomForestClassifier(
    n_estimators=100,    # number of trees
    max_depth=4,         # tree depth limit for simplicity
    random_state=2
)
rf_model.fit(X_train, y_train)

# Predictions
rf_predictions = rf_model.predict(X_test)

# Evaluate Model
print("===== Random Forest Evaluation =====")
print("\nConfusion Matrix:")
print(pd.DataFrame(confusion_matrix(y_test, rf_predictions),
                   index=['Actual Not Approved', 'Actual Approved'],
                   columns=['Predicted Not Approved', 'Predicted Approved']))

print("\nClassification Report:")
print(classification_report(y_test, rf_predictions))
accuracy = accuracy_score(y_test, rf_predictions)
print(f"Accuracy: {accuracy:.4f}")

# Feature Importance
rf_features = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nTop Feature Importances:")
print(rf_features.head(10))

# Optional Cross-Validation
rf_cv_score = cross_val_score(rf_model, X, y, cv=5).mean()
print(f"\nCross-Validation Accuracy: {rf_cv_score:.4f}")

# Feature Importance Visualization
plt.figure(figsize=(10,6))
plt.barh(rf_features['Feature'].head(10), rf_features['Importance'].head(10))
plt.gca().invert_yaxis()
plt.title("Top 10 Feature Importances - Random Forest")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()

print("\n Random Forest Model Training Complete.")


# === logging experiement results to decision_tree.log ===
csv_file = "Learning_Model_experiments.csv"
header = ["Timestamp", "Model", "Max Depth", "random_state", "Accuracy", "Cross-Validation Score"]

with open(csv_file, mode='a', newline='') as file:
    writer = csv.writer(file)
    if file.tell() == 0:
        writer.writerow(header)
    writer.writerow([
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Random Forest",
        4,  # max_depths
        2, #random_state
        f"{accuracy:.4f}",
        f"{rf_cv_score:.4f}"
    ])
