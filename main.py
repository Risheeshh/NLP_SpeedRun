# ============================
# NLP ASSIGNMENT FINAL CODE
# ============================

import pandas as pd
import numpy as np
import re
import nltk
import spacy
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics.pairwise import cosine_similarity

# ============================
# DOWNLOAD NLTK DATA
# ============================

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng') # Added this line
nltk.download('maxent_ne_chunker')
nltk.download('maxent_ne_chunker_tab') # Added this line to fix the LookupError
nltk.download('words')
nltk.download('omw-1.4')
nltk.download('punkt_tab')


# ============================
# TEXT PREPROCESSING
# ============================

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return ' '.join(words)

# ============================
# Q1: SENTIMENT ANALYSIS
# ============================

def sentiment_analysis():
    print("\n--- Q1: Sentiment Analysis ---")

    df = pd.read_csv("/content/sample_data/product_reviews.csv")
    df['cleaned'] = df['review_text'].apply(preprocess_text)
    df['label'] = df['sentiment'].map({'Positive':1, 'Negative':0})

    X = df['cleaned']
    y = df['label']

    vectorizer = TfidfVectorizer()
    X_tfidf = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.2, random_state=42, stratify=y
    )

    model = MultinomialNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Confusion matrix visualization
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title("Confusion Matrix")
    plt.show()

    return vectorizer, model

# ============================
# Q2: WORD SIMILARITY
# ============================

def word_similarity():
    print("\n--- Q2: Word Similarity ---")

    # !python -m spacy download en_core_web_md
    nlp = spacy.load("en_core_web_md")

    pairs = [("king","queen"), ("doctor","nurse"), ("car","tree")]

    for w1, w2 in pairs:
        sim = nlp(w1).similarity(nlp(w2))
        print(f"{w1} - {w2}: {sim:.4f}")

# ============================
# Q3: NAMED ENTITY RECOGNITION
# ============================

def named_entity_recognition():
    print("\n--- Q3: Named Entity Recognition ---")

    text = """On Tuesday, Sundar Pichai, the CEO of Google, announced a new lab in London, United Kingdom. Elon Musk said Tesla may expand in Texas."""

    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    tree = ne_chunk(pos_tags)

    print(tree)

# ============================
# Q4: CHATBOT
# ============================

def chatbot(vectorizer, model):
    print("\n--- Q4: Chatbot ---")

    # Expanded intents
    qa_pairs = {
        "what is nlp": "NLP stands for Natural Language Processing. It enables computers to understand and process human language.",
        "define nlp": "Natural Language Processing is a field of AI focused on interaction between computers and human language.",
        "what is ai": "Artificial Intelligence refers to systems that can perform tasks requiring human intelligence.",
        "applications of ai": "AI is used in chatbots, recommendation systems, fraud detection, and self-driving cars.",
        "what is machine learning": "Machine Learning is a subset of AI where systems learn patterns from data.",
        "difference between ai and ml": "AI is a broader concept, while ML is a subset that focuses on learning from data.",
        "what is deep learning": "Deep Learning uses neural networks with multiple layers to learn complex patterns.",
        "what is tf idf": "TF-IDF is a technique that assigns importance to words based on their frequency in a document and across documents.",
        "what is word embedding": "Word embeddings represent words as vectors capturing semantic meaning.",
        "what is ner": "Named Entity Recognition identifies entities like persons, locations, and organizations in text.",
        "what is chatbot": "A chatbot is a program designed to simulate conversation with users.",
        "what is sentiment analysis": "Sentiment analysis determines whether text expresses positive, negative, or neutral sentiment."
    }

    corpus = list(qa_pairs.keys())
    responses = list(qa_pairs.values())

    qa_vectorizer = TfidfVectorizer()
    qa_matrix = qa_vectorizer.fit_transform(corpus)

    while True:
        user_input = input("\nYou: ").strip().lower()

        # Exit condition
        if user_input in ["exit", "quit"]:
            print("Chatbot: Goodbye! 👋")
            break

        # Greeting handling
        if any(greet in user_input for greet in ["hi", "hello", "hey"]):
            print("Chatbot: Hello! Ask me anything about NLP or AI, or analyze a review.")
            continue

        # Sentiment analysis feature
        if user_input.startswith("analyze this review:"):
            review = user_input.replace("analyze this review:", "").strip()
            cleaned = preprocess_text(review)
            vec = vectorizer.transform([cleaned])
            pred = model.predict(vec)[0]
            sentiment = "Positive" if pred == 1 else "Negative"
            print(f"Chatbot: Sentiment = {sentiment}")
            continue

        # NLP preprocessing for intent matching
        cleaned_input = preprocess_text(user_input)
        user_vec = qa_vectorizer.transform([cleaned_input])

        sim = cosine_similarity(user_vec, qa_matrix)
        idx = sim.argmax()
        confidence = sim[0][idx]

        # Response logic
        if confidence > 0.35:
            print(f"Chatbot: {responses[idx]}")
            print(f"(Confidence: {confidence:.2f})")
        else:
            print("Chatbot: I'm not sure about that. Try asking about NLP, AI, or sentiment analysis.")
# ============================
# MAIN FUNCTION
# ============================

if __name__ == "__main__":
    vec, model = sentiment_analysis()
    word_similarity()
    named_entity_recognition()
    chatbot(vec, model)