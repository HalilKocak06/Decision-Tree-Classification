# Decision Tree Classification
This repository demonstrates a Decision Tree Classification model to predict user behavior based on a social network advertisement dataset. The project uses scikit-learn for machine learning, pandas for data manipulation, and matplotlib for visualization.

## Project Overview
The goal of this project is to predict whether a user will purchase a product based on their age and estimated salary. The decision tree classifier is trained and evaluated using the dataset Social_Network_Ads.csv.

## Dataset
The dataset Social_Network_Ads.csv contains the following columns:

Age: User's age.
Estimated Salary: User's salary estimation.
Purchased: Whether the user purchased the product (1: Yes, 0: No).
Project Workflow
Import Libraries: Import essential libraries like numpy, pandas, matplotlib, and scikit-learn.
Load Dataset: Load the dataset using pandas.
Split Dataset: Split the data into training and test sets.
Feature Scaling: Apply feature scaling to normalize the feature values.
Train the Model: Train a decision tree classifier using the training set.
Make Predictions: Use the model to predict test set results and make single predictions.
Evaluate the Model: Generate a confusion matrix and calculate the accuracy of the model.
Visualize the Results: Visualize the decision boundary for both training and test sets.


## Dependencies
To run this project, ensure you have the following Python libraries installed:

numpy
pandas
matplotlib
scikit-learn
You can install them using pip:

pip install numpy pandas matplotlib scikit-learn
How to Run
Clone the repository:

git clone https://github.com/your-username/decision-tree-classification.git
cd decision-tree-classification
Ensure you have the dataset file Social_Network_Ads.csv in the project directory.

## Run the Python script:

python decision_tree_classification.py

The results will include the accuracy score, confusion matrix, and visualizations for both the training and test sets.

## Code Overview
Training the Model
python

from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)

## Making Predictions
python

y_pred = classifier.predict(X_test)

## Model Evaluation
from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

## Visualizing the Results

plt.scatter(...)
plt.contourf(...)
plt.show()

## Results
Confusion Matrix: The confusion matrix is printed after prediction, providing a summary of the classifier's performance.
Accuracy: The accuracy score of the model is also displayed to evaluate its performance on the test set.
