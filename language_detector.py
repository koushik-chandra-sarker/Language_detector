# make a function that detects the language of a text from my previous coplot history

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

dataset = pd.read_csv('data/labeled/final_dataset.csv')
x_train, x_test, y_train, y_test = train_test_split(dataset["text"], dataset.language, test_size=0.20, random_state=1)
cv = CountVectorizer()
x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)
MNB = MultinomialNB()
MNB.fit(x_train, y_train)
print("MNB: ", MNB.score(x_test, y_test))
LR = LogisticRegression(max_iter=200)
LR.fit(x_train, y_train)
print("LR: ", LR.score(x_test, y_test))
DT = DecisionTreeClassifier()
DT.fit(x_train, y_train)
print("DT: ", DT.score(x_test, y_test))
KNN = KNeighborsClassifier(n_neighbors=3)
KNN.fit(x_train, y_train)
print("KNN: ", KNN.score(x_test, y_test))
SVM = SVC(kernel='linear')
SVM.fit(x_train, y_train)
print("SVM: ", SVM.score(x_test, y_test))

# def language_detector(text):
#     text = cv.transform([text])
#     return MNB.predict(text)
# print(language_detector("আমি ভালো আছি"))
