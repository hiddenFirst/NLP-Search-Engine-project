import os
import re
import webbrowser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import nltk

# Download resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

# Path
FOLDER_PATH = 'SimpleText_auto'

# Stop word list
STOP_WORDS = set(stopwords.words('english'))

def load_documents(folder_path):
    """Load all articles in a folder"""
    documents = {}
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                documents[file_name] = f.read()
    return documents

def preprocess_text(text):
    """Clean the text and remove stop words"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    words = word_tokenize(text)
    words = [word for word in words if word not in STOP_WORDS]  # Remove stop words
    return ' '.join(words)

def build_tfidf_index(documents):
    """Building an index using TF-IDF"""
    vectorizer = TfidfVectorizer()
    doc_names = list(documents.keys())
    doc_texts = [preprocess_text(text) for text in documents.values()]
    tfidf_matrix = vectorizer.fit_transform(doc_texts)
    return vectorizer, tfidf_matrix, doc_names

def search_query(query, vectorizer, tfidf_matrix, doc_names, top_n=10):
    """Search query and return relevant documents"""
    query_vector = vectorizer.transform([preprocess_text(query)])
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    ranked_indices = similarities.argsort()[::-1][:top_n]
    results = [(doc_names[i], similarities[i]) for i in ranked_indices]
    return results

def display_doc(doc_name):
    """Open text file from corpus"""
    full_path = os.path.abspath(f"SimpleText_auto/{doc_name}")
    webbrowser.open(f"file:///{full_path}")

def main():
    # Step 1: Load all Document
    print("Loading Document...")
    documents = load_documents(FOLDER_PATH)
    print(f"Success loading {len(documents)} documents.")

    # Step 2: Building a TF-IDF Index
    print("Building TF-IDF index...")
    vectorizer, tfidf_matrix, doc_names = build_tfidf_index(documents)
    print("Index building complete!")

    # Step 3: Enter search mode
    print("Search engine started! Type 'exit' to exit.")
    while True:
        query = input("Please enter your search query: ")
        if query.lower() == 'exit':
            print("Thanks for using, the search engine has been closed!")
            break
        results = search_query(query, vectorizer, tfidf_matrix, doc_names)
        print("Search results:")
        for i, (doc_name, score) in enumerate(results):
            with open(f'{FOLDER_PATH}/{doc_name}') as doc:
                title = doc.readline()
                print(f"{i+1}. Title: {title}", end = "")
                print(f"- Similarity: {score:.4f}")

        # Step 4: Enter view mode
        print("View mode started! Hit 'enter' to exit.")
        while True:
            query = input("Enter the number of the document to view: ")
            if not query:
                break
            display_doc(results[int(query) - 1][0])

if __name__ == "__main__":
    main()
