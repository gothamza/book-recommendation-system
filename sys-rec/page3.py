# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Initialize Elasticsearch connection
from elasticsearch import Elasticsearch
es = Elasticsearch("https://2eb9-102-97-167-182.ngrok-free.app", timeout=60, max_retries=10, retry_on_timeout=True)

# CSS for styling the images and links
st.markdown("""
    <style>
    .result-table { 
        margin-top: 20px; 
        border-collapse: collapse;
        width: 100%; 
    }
    .result-table td, .result-table th {
        border: 1px solid #ddd;
        padding: 8px;
        text-align: center;
    }
    .result-table tr:nth-child(even){background-color: #f2f2f2;}
    .result-table th {
        padding-top: 12px;
        padding-bottom: 12px;
        background-color: #4CAF50;
        color: white;
    }
    .cover-img {
        width: 60px;
        height: auto;
    }
    .book-link {
        color: #1f77b4;
        font-weight: bold;
        text-decoration: none;
    }
    .book-link:hover {
        color: #d62728;
    }
    </style>
""", unsafe_allow_html=True)

# Load data and models
@st.cache_data
def load_data():
    return pd.read_csv('df_books_users_processed.csv.gz')

@st.cache_resource
def load_models():
    with open("vectorizer_title.pkl", "rb") as p:
        vectorizer_title_fc = pickle.load(p)
    with open("tfidf_title.pkl", "rb") as p:
        tfidf_title_fc = pickle.load(p)
    return vectorizer_title_fc, tfidf_title_fc

df_books_processed = load_data()
vectorizer_title_fc, tfidf_title_fc = load_models()

# Helper functions for displaying URLs and images
def make_clickable(url):
    return f'<a target="_blank" href="{url}">More Info</a>'

def show_image(url):
    return f'<img src="{url}" class="cover-img">'

# Function for top 50 similar books
def top_50_similar_title_books(query, vectorizer):
    processed = re.sub("[^a-zA-Z0-9 ]", "", query.lower())
    query_vec = vectorizer.transform([processed])
    similarity = cosine_similarity(query_vec, tfidf_title_fc).flatten()
    indices = np.argpartition(similarity, -50)[-50:]
    results = df_books_processed.iloc[indices].sort_values(by="book_average_rating", ascending=False)
    results['Cover Page'] = results['cover_page'].apply(show_image)
    results['Book URL'] = results['book_url'].apply(make_clickable)
    return results[['title_without_series', 'book_average_rating', 'Cover Page', 'Book URL']].head(5)

# Sidebar menu
st.sidebar.title("📚 Book Application")
menu_option = st.sidebar.radio("Select a function:", ["Autocomplete Book Search", "Top Concise Books", "Find Similar Books by Title"])

# Autocomplete Book Search functionality
if menu_option == "Autocomplete Book Search":
    st.title("📚 Real-Time Book Autocomplete System")
    st.markdown("Start typing a book title to receive suggestions in real time! 🎉")

    # Input box for real-time autocomplete
    user_input = st.text_input("🔍 Search for a book:", "")

    # Real-time search execution upon input change
    if user_input:
        # Query Elasticsearch for suggestions
        response = es.search(
            index="books",
            body={
                "suggest": {
                    "book-title-suggest": {
                        "prefix": user_input,
                        "completion": {
                            "field": "title_without_series",
                            "fuzzy": {
                                "fuzziness": 2
                            }
                        }
                    }
                }
            }
        )
        
        # Extract suggestions from the response
        suggestions = response['suggest']['book-title-suggest'][0]['options']
        
        # Process suggestions into a DataFrame
        data = []
        for suggestion in suggestions:
            source = suggestion['_source']
            data.append({
                'Title': source['title_without_series'].title(),
                'Publication Year': source['publication_year'],
                'Publisher': source['publisher'],
                'Average Rating': source['book_average_rating'],
                'Cover Page': show_image(source['cover_page']),
                'Book URL': make_clickable(source['book_url'])
            })
        
        # Display the DataFrame if results exist
        if data:
            results = pd.DataFrame(data)[['Title', 'Publication Year', 'Publisher', 'Average Rating', 'Cover Page', 'Book URL']]
            st.markdown('<table class="result-table">' + results.to_html(escape=False, index=False) + '</table>', unsafe_allow_html=True)
        else:
            st.warning("No results found. Please try a different search term.")
    else:
        st.info("Type in the box above to begin your search!")

# Top Concise Books functionality
elif menu_option == "Top Concise Books":
    st.title("📚 Top Concise Books Recommendation System")
    st.markdown("Displaying books with less than 300 pages, high ratings, and many reviews.")

    # Filter top concise books if data is available
    if not df_books_processed.empty:
        top_books = df_books_processed[
            (df_books_processed['num_pages'] <= 300) & 
            (df_books_processed['ratings_count'] > 3000) & 
            (df_books_processed['book_average_rating'] >= 4.5)
        ].sort_values(by='book_average_rating', ascending=False).head(5)
        
        if not top_books.empty:
            top_books['Cover Page'] = top_books['cover_page'].apply(show_image)
            top_books['Book URL'] = top_books['book_url'].apply(make_clickable)

            st.markdown('<table class="result-table">' + top_books[['title_without_series', 'num_pages', 'book_average_rating', 'Cover Page', 'Book URL']].to_html(escape=False, index=False) + '</table>', unsafe_allow_html=True)
        else:
            st.warning("No books found matching the criteria.")
    else:
        st.error("Failed to load data. Please check the file path or data format.")

# Find Similar Books by Title functionality
elif menu_option == "Find Similar Books by Title":
    st.title("📖 Find Similar Books by Title")
    st.markdown("Enter a book title to find similar books based on text similarity.")

    # Input box for finding similar books
    book_title = st.text_input("🔍 Enter book title:", "")

    if book_title:
        similar_books = top_50_similar_title_books(book_title, vectorizer_title_fc)
        
        if not similar_books.empty:
            st.markdown('<table class="result-table">' + similar_books.to_html(escape=False, index=False) + '</table>', unsafe_allow_html=True)
        else:
            st.warning("No similar books found.")
