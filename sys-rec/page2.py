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

def load_data2():
    return pd.read_csv('df_final_review.csv.gz')

def load_models():
    with open("vectorizer_title.pkl", "rb") as p:
        vectorizer_title_fc = pickle.load(p)
    with open("tfidf_title.pkl", "rb") as p:
        tfidf_title_fc = pickle.load(p)

    with open("vectorizer_description.pkl", "rb") as p:
        vectorizer_description = pickle.load(p)

    with open("tfidf_description.pkl", "rb") as p:
        tfidf_description = pickle.load(p)
        
    with open("vectorizer_review.pkl", "rb") as p:
        vectorizer_review = pickle.load(p)

    with open("tfidf_review.pkl", "rb") as p:
        tfidf_review = pickle.load(p)
    
    return vectorizer_title_fc, tfidf_title_fc ,vectorizer_description ,tfidf_description ,vectorizer_review ,tfidf_review


df_books_processed = load_data()
# df_final_review = load_data2()
# vectorizer_title_fc, tfidf_title_fc ,vectorizer_description ,tfidf_description,vectorizer_review ,tfidf_review= load_models()

# Helper functions for displaying URLs and images
def make_clickable(url):
    return f'<a target="_blank" href="{url}">More Info</a>'

def show_image(url):
    return f'<img src="{url}" class="cover-img">'

# Recommendation functions
def similar_user_df(df_books_users, user_id):
    df_liked_books = df_books_users[df_books_users['user_id'] == user_id]
    liked_books = set(df_liked_books['book_id'])
    top_5_liked_books = df_liked_books.sort_values(by='rating', ascending=False)['book_id'][:5]
    similar_user = df_books_users[(df_books_users['book_id'].isin(top_5_liked_books)) & (df_books_users['rating'] > 4)]['user_id']
    data = df_books_users[(df_books_users['user_id'].isin(similar_user))][['user_id', 'book_id', 'rating', 'ratings_count', 'title_without_series', 'book_average_rating', 'book_url', 'cover_page']]
    return data, liked_books

def popular_recommendation(recs, liked_books):
    all_recs = recs["book_id"].value_counts()
    all_recs = all_recs.to_frame().reset_index()
    all_recs.columns = ["book_id", "book_count"]
    all_recs = all_recs.merge(recs, how="inner", on="book_id")
    all_recs["score"] = all_recs["book_count"] * (all_recs["book_count"] / all_recs["ratings_count"])
    popular_recs = all_recs.sort_values("score", ascending=False)
    popular_recs_unbiased = popular_recs[~popular_recs["book_id"].isin(liked_books)].drop_duplicates(subset=['title_without_series'])
    popular_recs_unbiased['Cover Page'] = popular_recs_unbiased['cover_page'].apply(show_image)
    popular_recs_unbiased['Book URL'] = popular_recs_unbiased['book_url'].apply(make_clickable)
    return popular_recs_unbiased[['title_without_series', 'book_average_rating', 'Cover Page', 'Book URL', 'ratings_count']].head(5)




# Définition des fonctions de formatage
def make_clickable(val):
    return f'<a href="{val}" target="_blank">{val}</a>'

def show_image(val):
    return f'<img src="{val}" width="60">'




def books_vedette():
    st.title("📚 Livres vedettes depuis la date spécifiée")

    # Request the date from the user through a text input box
    user_input = st.text_input("Enter the date in 'YYYY-MM-DD' format:")

    # Verify the date format
    try:
        start_date = pd.to_datetime(user_input, format='%Y-%m-%d')
    except ValueError:
        st.error("Invalid date format. Please enter the date in 'YYYY-MM-DD' format.")
        return

    # Ensure 'read_at' is datetime and timezone-naive
    # Convert to datetime with error coercion to handle non-datetimelike values
    # Convert 'read_at' to datetime, setting invalid formats to NaT
    # Convert 'read_at' to datetime, setting invalid formats to NaT
    df_books_processed['read_at'] = pd.to_datetime(df_books_processed['read_at'], errors='coerce')

    # Convert all entries to pd.Timestamp and remove timezone if present
    df_books_processed['read_at'] = df_books_processed['read_at'].apply(
        lambda x: pd.Timestamp(x).tz_localize(None) if x is not pd.NaT and x.tzinfo else x
    )
        # Drop timezone info if present
    # Make 'start_date' timezone-naive
    start_date = start_date.tz_localize(None)

    # Filter books read after the specified date
    books_after_date = df_books_processed[df_books_processed['read_at'] >= start_date]

    # Ensure 'rating' is numeric
    df_books_processed['rating'] = pd.to_numeric(df_books_processed['rating'], errors='coerce')

    # Calculate the number of votes for each book
    top_books = books_after_date.groupby('book_id').agg({
        'rating': 'mean',   # Average rating
        'read_at': 'count'  # Count of how many times the book was read
    }).reset_index()

    # Rename columns for clarity
    top_books.rename(columns={'rating': 'average_rating', 'read_at': 'n_reads'}, inplace=True)

    # Filter for books that have been read at least 3 times
    top_books = top_books[top_books['n_reads'] >= 3]

    # Merge with additional book info
    top_books = top_books.merge(df_books_processed[['book_id', 'title_without_series', 'book_url', 'cover_page']].drop_duplicates(), on='book_id')

    # Sort books by average rating and number of reads
    top_books = top_books.sort_values(by=['average_rating', 'n_reads'], ascending=False)

    # Display results
    if top_books.empty:
        st.warning("No featured books found since the specified date.")
    else:
        st.markdown("### Featured Books Since the Specified Date:")
        top_books['Book URL'] = top_books['book_url'].apply(make_clickable)
        top_books['Cover Page'] = top_books['cover_page'].apply(show_image)
        display_columns = ['title_without_series', 'average_rating', 'n_reads', 'Book URL', 'Cover Page']
        st.markdown('<table class="result-table">' + top_books[display_columns].to_html(escape=False, index=False) + '</table>', unsafe_allow_html=True)



# Sidebar menu
st.sidebar.title("📚 Book Application")
menu_option = st.sidebar.radio("Select a function:", ["Autocomplete Book Search", "Top Concise Books", "Find Similar Books by Title", "Books Where User Choice's Match","Books having similar description","Books having similar review","Books vedettes"])

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

    # # Input box for finding similar books
    # book_title = st.text_input("🔍 Enter book title:", "")

    # if book_title:
    #     similar_books = top_50_similar_title_books(book_title, vectorizer_title_fc)
        
    #     if not similar_books.empty:
    #         st.markdown('<table class="result-table">' + similar_books.to_html(escape=False, index=False) + '</table>', unsafe_allow_html=True)
    #     else:
    #         st.warning("No similar books found.")

# Books Where User Choice's Match functionality
elif menu_option == "Books Where User Choice's Match":
    st.title("📚 Books Where User Choice's Match")
    st.markdown("Enter a user ID to get recommendations based on similar users' preferences.")

    # Input box for user ID
    user_id = st.text_input("🔍 Enter User ID:", "")

    if user_id:
        # Retrieve similar user data and liked books
        recs, liked_books = similar_user_df(df_books_processed, user_id)
        
        if not recs.empty:
            # Generate popular recommendations
            popular_recs = popular_recommendation(recs, liked_books)
            st.markdown('<table class="result-table">' + popular_recs.to_html(escape=False, index=False) + '</table>', unsafe_allow_html=True)
        else:
            st.warning("No matching recommendations found for the entered user.")
elif menu_option == "Books having similar description" :
    st.title("📚 Books having similar description")
    st.markdown("Enter a Book description to get recommendations based on Books similar description.")

    # # Input box for user ID
    # query = st.text_input("🔍 Enter Book description:", "")
    # if query:
       
    #     # Generate popular recommendations
    #     popular_recs = search(query,vectorizer_description)
    #     st.markdown('<table class="result-table">' + popular_recs.to_html(escape=False, index=False) + '</table>', unsafe_allow_html=True)
    # else:
    #     st.warning("No similar books found.")
    
elif menu_option == "Books having similar review" :
    st.title("📚 Books having similar review")
    st.markdown("Enter a Book ID to get recommendations based on Books similar review.")

    # # Input box for user ID
    # query = st.text_input("🔍 Enter Book ID:", "")
    # if query:
       
    #     # Generate popular recommendations
    #     popular_recs = review_similarity(query)
    #     st.markdown('<table class="result-table">' + popular_recs.to_html(escape=False, index=False) + '</table>', unsafe_allow_html=True)
    # else:
    #     st.warning("No similar books found.")

# Adding to the sidebar menu
elif menu_option == "Books vedettes":
    books_vedette()
