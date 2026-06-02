# ==========================================
# AI Learning ChatBot
# TF-IDF + Logistic Regression + RapidFuzz
# ==========================================

import pandas as pd
from datetime import datetime

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
# Topic List
# ==========================================

topics = list(df["question"].unique())

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
# ChatBot Start
# ==========================================

print("=" * 60)
print("          AI Learning ChatBot")
print("          Type 'exit' to quit")
print("=" * 60)

print("\nBot: Hello! How can I help you? 😊")

# ==========================================
# Chat Loop
# ==========================================

while True:

    user_input = input("\nYou: ").strip()

    if user_input.lower() == "exit":

        print("\nBot: Goodbye! 👋")

        break

    # ======================================
    # Clean Input
    # ======================================

    cleaned_input = clean_text(
        user_input
    )

    cleaned_topics = [
        clean_text(topic)
        for topic in topics
    ]

    # ======================================
    # Fuzzy Matching
    # ======================================

    match = process.extractOne(
        cleaned_input,
        cleaned_topics
    )

    # ======================================
    # Unknown Topic Detection
    # ======================================

    if not match or match[1] < 85:

        current_time = datetime.now().strftime(
            "%H:%M:%S"
        )

        print(
            f"[{current_time}] Bot: I don't know about this topic yet."
        )

        with open(
            "unknown_questions.txt",
            "a",
            encoding="utf-8"
        ) as file:

            file.write(
                user_input + "\n"
            )

        continue

    # ======================================
    # Find Original Topic
    # ======================================

    best_match = match[0]

    for topic in topics:

        if clean_text(topic) == best_match:

            user_input = topic

            break

    # ======================================
    # ML Prediction
    # ======================================

    user_vector = vectorizer.transform(
        [user_input]
    )

    prediction = model.predict(
        user_vector
    )[0]

    confidence = model.predict_proba(
        user_vector
    ).max()

    current_time = datetime.now().strftime(
        "%H:%M:%S"
    )

    # ======================================
    # Display Answer
    # ======================================

    print(
        f"[{current_time}] Bot: {prediction}"
    )

    print(
        f"(Confidence: {confidence:.2f})"
    )

    # ======================================
    # Save Chat History
    # ======================================

    with open(
        "chat_history.txt",
        "a",
        encoding="utf-8"
    ) as file:

        file.write(
            f"\n[{datetime.now()}]\n"
        )

        file.write(
            f"You: {user_input}\n"
        )

        file.write(
            f"Bot: {prediction}\n"
        )

# ==========================================
# End Program
# ==========================================