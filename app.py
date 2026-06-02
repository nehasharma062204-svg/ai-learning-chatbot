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
# TF-IDF Vectorization
# ==========================================

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 3)
)

X_vectorized = vectorizer.fit_transform(X)

# ==========================================
# Train Logistic Regression
# ==========================================

model = LogisticRegression(
    max_iter=3000
)

model.fit(X_vectorized, y)

# ==========================================
# Text Cleaning Function
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
# Create Topics Automatically
# ==========================================

main_topics = []

for question in df["question"]:

    cleaned = clean_text(question)

    if cleaned not in main_topics:

        main_topics.append(cleaned)

# ==========================================
# ChatBot Response Function
# ==========================================

def get_response(user_input):

    user_input = user_input.lower().strip()

    # --------------------------------------
    # Greeting Handling
    # --------------------------------------

    greetings = [
        "hi",
        "hello",
        "hey",
        "hii",
        "hiii",
        "good morning",
        "good afternoon",
        "good evening"
    ]

    if user_input in greetings:

        return (
            "Hello! How can I help you? 😊",
            1.0
        )

    # --------------------------------------
    # Exit Handling
    # --------------------------------------

    if user_input in [
        "exit",
        "quit",
        "bye"
    ]:

        return (
            "Goodbye! 👋",
            1.0
        )

    # --------------------------------------
    # Clean Text
    # --------------------------------------

    cleaned_input = clean_text(
        user_input
    )

    # --------------------------------------
    # Fuzzy Matching
    # --------------------------------------

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

    # --------------------------------------
    # Convert To Vector
    # --------------------------------------

    user_vector = vectorizer.transform(
        [matched_topic]
    )

    # --------------------------------------
    # Predict Answer
    # --------------------------------------

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

        if confidence == 0:

            st.warning(answer)

        else:

            st.success(answer)

            # Confidence sirf ML answers ke liye

            if answer not in [
                "Hello! How can I help you? 😊",
                "Goodbye! 👋"
            ]:

                st.caption(
                    f"Confidence: {confidence:.2f}"
                )