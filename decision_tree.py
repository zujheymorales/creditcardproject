#Libraries
import pandas as pd 
import matplotlib.pyplot as plt   
import logging
import csv
from sklearn.tree import DecisionTreeClassifier, plot_tree  
from sklearn.model_selection import train_test_split, cross_val_score  
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score  
from datetime import datetime

#====Set up logging===
logging.basicConfig(
    filename='decision_tree.log', 
    level = logging.INFO,
    format = '%(asctime)s - %(levelname)s - %(message)s'
)
logging.info(" === New Run === ")


try:
    #load dataset
    df = pd.read_csv('clean_dataset.csv')
    logging.info("Dataset loaded Successfully")

    #drop all non-integer columns
    df = df.drop(columns=['Industry', 'Ethnicity', 'Citizen'])  

    #split the data into features and target variable
    X = df.drop(columns=['Approved'])
    y = df['Approved']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
    logging.info("Data split into training and testing sets")

    #====== train the model and evaluate performance ====
    model = DecisionTreeClassifier(max_depth=4, random_state=10)

    #Fit the model on the training data
    model.fit(X_train ,y_train)

    #===== predictions on the testing data ====
    y_predictions = model.predict(X_test)

    features = pd.DataFrame(model.feature_importances_, index=X.columns)
    features.head(12)

    #Evaluate accuracy of the model
    accuracy = model.score(X_test, y_test)

    #cross validation
    df_cv_score = cross_val_score(model, X, y, cv=5).mean()

    #print analysis
    print("===== Decision Tree Evaluation =====")

    print("\nConfusion Matrix:")
    print(pd.DataFrame(confusion_matrix(y_test, y_predictions),
                        index=['Actual Not Approved', 'Actual Approved'],
                        columns=['Predicted Not Approved', 'Predicted Approved']))
    print("\nClassification Report:")
    print(classification_report(y_test, y_predictions))
    print(f"Accuracy: {accuracy:.4f}")
    print(f"\nCross-Validation Accuracy: {df_cv_score:.4f}")

    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"Cross-Validation: {df_cv_score:.4f}")


    #Visualize the decision tree
    plt.figure(figsize=(10,6))
    plot_tree(model, filled=True, feature_names=X.columns, 
            class_names=['Not Approved', 'Approved'], max_depth=4)
    plt.title("Decision Tree Visualization")
    plt.show()
    logging.info("Evaluation saved to decision_tree.log")


    # ==== Save Experiement to CSV ====
    csv_file = "decision_tree_experiments.csv"

    header = [
        "Timestamp", "Model", "Max Depth","random_state", "Accuracy", "Cross-Validation Score"
    ]
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(header) 
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Decision Tree",
            4,
            10,
            f"{accuracy:.4f}",
            f"{df_cv_score:.4f}"
        ])
        logging.info( "Results saved to decision_tree_experiments.csv")

except Exception as e:
    logging.error(f"Error occurred:, {e}")
    print(f"Oops! there was an error: {e}")