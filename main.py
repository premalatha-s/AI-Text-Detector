import pandas as pd
import string

data = pd.read_csv("dataset.csv")

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

data['text'] = data['text'].apply(preprocess)

print("Dataset Loaded Successfully ")
print(data.head())

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(data['text'])
y = data['label']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

# Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

lr_pred = lr_model.predict(X_test)
nb_pred = nb_model.predict(X_test)

print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_pred))
print("Naive Bayes Accuracy:", accuracy_score(y_test, nb_pred))

def decision_layer(confidence):
    if confidence >= 0.80:
        return "Acceptable"
    elif confidence >= 0.60:
        return "Needs Review"
    else:
        return "Likely AI-generated / Uncertain"
    
def predict_text(text, model):
     text = preprocess(text)
     vector = vectorizer.transform([text])
    
     prediction = model.predict(vector)[0]
     probability = model.predict_proba(vector).max()
    
     decision = decision_layer(probability)
    
     print("\n--- RESULT ---")
     print("Text:", text)
     print("Prediction:", prediction)
     print("Confidence:", round(probability, 2))
     print("Decision:", decision)

user_input = input("Enter text: ")
predict_text(user_input, lr_model)