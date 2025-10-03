#Decision Tree Classifier
import pandas as pd #data manipulation
from sklearn.tree import DecisionTreeClassifier     #decision tree model
from sklearn.model_selection import train_test_split #train test split
from sklearn.metrics import confusion_matrix         #confusion matrix
from sklearn.metrics import classification_report  #classification report
#from sklearn.metrics import accuracy_score          #accuracy score

#load dataset
df = pd.read_csv('clean_dataset.csv')


df = df.drop(columns=['Industry', 'Ethnicity', 'Citizen']) #drop all non-integer columns

X = df.drop(columns=['Approved'])
y = df['Approved']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10, test_size=0.2 )

#train the model and evaluate preformance
model = DecisionTreeClassifier()

#Fit the model on the training data
model.fit(X_train ,y_train)

#made the predictions on the testing data
y_predictions = model.predict(X_test)

print(confusion_matrix(y_test, y_predictions))

print(classification_report(y_test, y_predictions))


features = pd.DataFrame(model.feature_importances_, index=X.columns)
features.head(14)

#Evalutate the model
#accuracy = model.score(x_test, y_test)

#print the accuracy of the model
#print(f"\Accuracy: {accuracy * 100:.2f}%")