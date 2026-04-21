import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load data
df = pd.read_csv("ACME-HappinessSurvey2020.csv")

# Separate features and label
X = df.drop(columns=["Y"])
y = df["Y"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
#print("Accuracy:", accuracy)

print(y.value_counts(normalize=True))

from sklearn.metrics import classification_report

y_pred = model.predict(X_test)
print("Classification Report Tool:\n", classification_report(y_test, y_pred))

#############################################

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

scores = cross_val_score(model, X, y, cv=5)

print("Scores:", scores)
print("Average accuracy:", scores.mean())

baseline = max(y.value_counts(normalize=True))
print("Baseline accuracy:", baseline)

from sklearn.metrics import confusion_matrix

y_pred = model.fit(X_train, y_train).predict(X_test)
print(confusion_matrix(y_test, y_pred))

y_pred = model.predict(X_test)
print("Classification Report Tool:\n", classification_report(y_test, y_pred))

##############################################
print("non-linear model")
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(max_depth=4)
scores = cross_val_score(model, X, y, cv=5)

print("Average accuracy:", scores.mean())

print("do any features matter?")
model.fit(X, y)
importances = model.feature_importances_

for name, val in zip(X.columns, importances):
    print(name, val)

y_pred = model.predict(X_test)
print("Classification Report Tool:\n", classification_report(y_test, y_pred))

########################################

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(max_depth=4)

scores = cross_val_score(model, X, y, cv=5)

print("Scores:", scores)
print("Mean:", scores.mean())
print("Std:", scores.std())

#y_pred = model.predict(X_test)
#print("Classification Report Tool:", classification_report(y_test, y_pred))

####################################SVM
print("\nTrying SVM.\n")

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

model = SVC(kernel='rbf', C=1, gamma='scale')

scores = cross_val_score(model, X_scaled, y, cv=5)

print("Scores:", scores)
print("Mean:", scores.mean())
print("Std:", scores.std())

from sklearn.model_selection import GridSearchCV

params = {
    "C": [0.1, 1, 10],
    "gamma": ["scale", 0.1, 1]
}

#y_pred = model.predict(X_test)
#print("Classification Report Tool:", classification_report(y_test, y_pred))

grid = GridSearchCV(SVC(kernel='rbf'), params, cv=5)
grid.fit(X_scaled, y)

print("Best score:", grid.best_score_)
print("Best params:", grid.best_params_)

#y_pred = model.predict(X_test)
#print("Classification Report Tool:", classification_report(y_test, y_pred))