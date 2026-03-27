from flask import Flask, render_template, request
import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

data = pd.read_csv("dataset.csv")

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

data['text'] = data['text'].apply(preprocess)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['text'])
y = data['label']


model = LogisticRegression()
model.fit(X, y)

def decision_layer(confidence):
    if confidence >= 0.80:
        return "Acceptable"
    elif confidence >= 0.60:
        return "Needs Review"
    else:
        return "Likely AI-generated"

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None

    if request.method == 'POST':
        text = request.form['text']
        processed = preprocess(text)
        vector = vectorizer.transform([processed])

        prediction = model.predict(vector)[0]
        confidence = model.predict_proba(vector).max()
        decision = decision_layer(confidence)

        word_count = len(text.split())

        if confidence >= 0.80:
            level = "High"
        elif confidence >= 0.60:
            level = "Medium"
        else:
            level = "Low"

        result = {
            "prediction": prediction,
            "confidence": round(confidence, 2),
            "decision": decision,
            "word_count": word_count,
            "level": level
        }

    return render_template('index.html', result=result)

if __name__ == "__main__":
    app.run(debug=True)