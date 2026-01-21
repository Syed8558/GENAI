import os
import json

def load_tickets(folder="../ticket_086"):
    tickets = []
    texts = []

    for file in os.listdir(folder):
        if file.endswith(".json"):
            with open(os.path.join(folder, file), "r") as f:
                data = json.load(f)
                text = data["title"] + " " + data["description"]
                tickets.append(data)
                texts.append(text)

    return tickets, texts

tickets, documents = load_tickets()
N = len(documents)

def tokenize(text):
    return text.lower().split()

tokenized_docs = [tokenize(doc) for doc in documents]

import math

doc_lengths = [len(doc) for doc in tokenized_docs]
avgDL = sum(doc_lengths) / N

k1 = 1.5
b = 0.75

# Vocabulary
vocab = set(word for doc in tokenized_docs for word in doc)

# Document Frequency
df = {}
for word in vocab:
    df[word] = sum(1 for doc in tokenized_docs if word in doc)

# IDF
idf = {}
for word in vocab:
    idf[word] = math.log((N - df[word] + 0.5) / (df[word] + 0.5) + 1)

def bm25_score(query, index):
    score = 0
    doc = tokenized_docs[index]

    for word in query:
        if word in doc:
            tf = doc.count(word)
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * doc_lengths[index] / avgDL)
            score += idf[word] * (numerator / denominator)

    return score


import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(documents)

def search_bm25(query):
    query_tokens = tokenize(query)
    scores = [bm25_score(query_tokens, i) for i in range(N)]
    best_idx = scores.index(max(scores))
    return tickets[best_idx], scores[best_idx]

def search_cosine(query):
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix)[0]
    best_idx = np.argmax(scores)
    return tickets[best_idx], scores[best_idx]

def search_euclidean(query):
    query_vec = vectorizer.transform([query])
    distances = euclidean_distances(query_vec, tfidf_matrix)[0]
    best_idx = np.argmin(distances)
    return tickets[best_idx], distances[best_idx]

def helpdesk_chatbot(user_query, method):
    if method == "bm25":
        ticket, score = search_bm25(user_query)
    elif method == "cosine":
        ticket, score = search_cosine(user_query)
    elif method == "euclidean":
        ticket, score = search_euclidean(user_query)
    else:
        raise ValueError("Invalid method")

    return {
        "best_match_ticket": ticket,
        "similarity_score": float(score),
        "method_used": method
    }

if __name__ == "__main__":
    print("\nIT Helpdesk Ticket Chatbot")
    print("Choose similarity method: bm25 | cosine | euclidean\n")

    while True:
        query = input("Describe your issue (or type exit): ")
        if query.lower() == "exit":
            break

        method = input("Method: ").lower()
        result = helpdesk_chatbot(query, method)

        print("\nBest Match (JSON):")
        print(json.dumps(result, indent=4))
        
        
        



