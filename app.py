import streamlit as st
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from rapidfuzz import process

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv("ml_chatbot_dataset.csv")

df["question"] = df["question"].str.lower().str.strip()

X = df["question"]
y = df["answer"]

# ==========================================
# TF-IDF
# ==========================================

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 3)
)

X_vectorized = vectorizer.fit_transform(X)

# ==========================================
# Logistic Regression
# ==========================================

model = LogisticRegression(
    max_iter=3000
)

model.fit(X_vectorized, y)

# ==========================================
# Main Topics
# ==========================================

main_topics = [
    "python",
    "variable",
    "datatype",
    "if statement",
    "for loop",
    "while loop",
    "loop",
    "break",
    "continue",
    "function",
    "return",
    "list",
    "tuple",
    "set",
    "dictionary",
    "class",
    "object",
    "constructor",
    "inheritance",
    "polymorphism",
    "encapsulation",
    "abstraction",
    "exception handling",
    "try except",
    "numpy",
    "pandas",
    "dataframe",
    "machine learning",
    "linear regression",
    "logistic regression",
    "decision tree",
    "random forest",
    "knn",
    "svm",
    "k means",
    "accuracy",
    "confusion matrix"
]

# ==========================================
# Text Cleaning
# ==========================================

def clean_text(text):

    text = text.lower().strip()

    prefixes = [
        "what is ",
        "define ",
        "explain ",
        "tell me about "
    ]

    for prefix in prefixes:

        if text.startswith(prefix):

            text = text.replace(
                prefix,
                "",
                1
            )

    return text

# ==========================================
# Response Function
# ==========================================

def get_response(user_input):

    cleaned_input = clean_text(
        user_input
    )

    # ----------------------------------
    # Fuzzy Matching
    # ----------------------------------

    match = process.extractOne(
        cleaned_input,
        main_topics
    )

    if not match or match[1] < 85:

        return (
            "I don't know about this topic yet.",
            0
        )

    matched_topic = match[0]

    # ----------------------------------
    # ML Prediction
    # ----------------------------------

    user_vector = vectorizer.transform(
        [matched_topic]
    )

    prediction = model.predict(
        user_vector
    )[0]

    confidence = model.predict_proba(
        user_vector
    ).max()

    return (
        prediction,
        confidence
    )

# ==========================================
# Streamlit UI
# ==========================================

st.set_page_config(
    page_title="AI Learning ChatBot",
    page_icon="🤖",
    layout="centered"
)

st.title(
    "🤖 AI Learning ChatBot"
)

st.write(
    "Ask me about Python, OOP, NumPy, Pandas and Machine Learning."
)

st.info(
    "Hello! How can I help you? 😊"
)

user_input = st.text_input(
    "Enter your question:"
)

if st.button("Send"):

    if user_input.strip():

        answer, confidence = get_response(
            user_input
        )

        if answer == "I don't know about this topic yet.":

            st.warning(answer)

        else:

            st.success(answer)

            st.caption(
                f"Confidence: {confidence:.2f}"
            )