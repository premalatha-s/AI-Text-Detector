# AI vs Human Text Detection System

##  Objective

The objective of this project is to build a Machine Learning system that can classify whether a given text is human-written or AI-generated.


##  Models Used

* Logistic Regression
* Naive Bayes


##  Features

* Text preprocessing (lowercase conversion, punctuation removal)
* TF-IDF vectorization
* Multiple machine learning models for classification
* Confidence-based intelligent decision system


##  Decision Logic

The system includes a rule-based decision layer:

* Confidence ≥ 0.80 →  Acceptable (High certainty)
* 0.60 ≤ Confidence < 0.80 →  Needs Review (Moderate certainty)
* Confidence < 0.60 →  Likely AI-generated / Uncertain


##  Dataset

A custom dataset is used containing both human-written and AI-generated text samples.


##  How to Run

1. Create virtual environment:

python -m venv venv

2. Activate virtual environment:

venv\Scripts\activate

3. Install dependencies:

pip install pandas scikit-learn flask

4. Run the application:

python app.py

5. Open browser and go to:

http://127.0.0.1:5000


##  Output

For any input text, the system provides:

* Prediction (Human / AI-generated)
* Confidence score
* Final decision (Acceptable / Needs Review / Likely AI-generated)


##  Key Highlights

* Works completely offline
* No external APIs used
* Lightweight and efficient
* Easy to deploy


##  Note

This is a basic Machine Learning model. The accuracy depends on the dataset size and quality.


##  Author

Premalatha S
