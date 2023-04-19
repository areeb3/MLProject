import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt

# Load the data from CSV file
data = pd.read_csv('fitness.csv')

print(data.head())

X = data.iloc[:, 1:-1] # Input features
y = data.iloc[:,-1]  # Target variable


# Split the data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=42) 
# Create a Random Forest classifier with 100 trees
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the Random Forest classifier on the training data
clf.fit(X_train, y_train)

# plt.plot(X_train,y_train)
# plt.show()

# plt.plot(X_test,y_test)
# plt.show()

#  Predict the target variable for the train  data
y_train_pred = clf.predict(X_train)
# Compute and display the accuracy score
acc1 = accuracy_score(y_train, y_train_pred)
print("Accuracy: for training data", acc1)

# Predict the target variable for the test data
y_pred = clf.predict(X_test)

# plt.plot(y_test,y_pred)
# plt.show()

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Display the confusion matrix and classification report
print("Confusion Matrix:")
print(cm)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Compute and display the accuracy score
acc = accuracy_score(y_test, y_pred)
print("Accuracy for testing data:", acc)


# # Plot the confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(np.unique(y)))
plt.xticks(tick_marks, np.unique(y), rotation=45)
plt.yticks(tick_marks, np.unique(y))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


x=["Very Low","Low","High","Very High"]
import seaborn as sns
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=x, yticklabels=x)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
