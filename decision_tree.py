#Decision Tree Classifier
import pandas as pd #data manipulation
import matplotlib.pyplot as plt                             #plotting library
from sklearn.tree import DecisionTreeClassifier, plot_tree  #decision tree and plot tree
from sklearn.model_selection import train_test_split        #train test split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score #model evaluation metrics

#load dataset
df = pd.read_csv('clean_dataset.csv')

#drop all non-integer columns
df = df.drop(columns=['Industry', 'Ethnicity', 'Citizen'])  

#split the data into features and target variable
X = df.drop(columns=['Approved'])
y = df['Approved']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

#train the model and evaluate preformance
model = DecisionTreeClassifier(max_depth=4, random_state=10)

#Fit the model on the training data
model.fit(X_train ,y_train)

#made the predictions on the testing data
y_predictions = model.predict(X_test)

features = pd.DataFrame(model.feature_importances_, index=X.columns)
features.head(14)

#Visualize the decision tree
plt.figure(figsize=(30,20))
plot_tree(model, filled=True, feature_names=X.columns, class_names=['Not Approved', 'Approved'], max_depth=4)
plt.title("Decision Tree Visualization")
plt.show(block=True)

#Evalutate the model
accuracy = model.score(X_test, y_test)

#print analysis
print(confusion_matrix(y_test, y_predictions))
print(classification_report(y_test, y_predictions))
print(accuracy_score(y_test, y_predictions))

