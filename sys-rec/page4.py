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

@st.cache_data
def load_data2():
    return pd.read_csv('df_final_review.csv.gz')

@st.cache_data
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

@st.cache_data
def load_models2():
    with open("le1.pkl", "rb") as p:
        le1 = pickle.load(p)
    with open("le4.pkl", "rb") as p:
        le4 = pickle.load(p)
    with open("le7.pkl", "rb") as p:
        le7 = pickle.load(p)
    with open("le9.pkl", "rb") as p:
        le9 = pickle.load(p)
    with open("clf_lr.pkl", "rb") as p:
        clf_lr = pickle.load(p)
    with open("vectorizer_title.pkl", "rb") as p:
        vectorizer_title = pickle.load(p)
    with open("norm.pkl", "rb") as p:
        norm = pickle.load(p)
    return clf_lr,vectorizer_title,le1,le4,le7,le9,norm

df_books_processed = load_data()
df_final_review = load_data2()

vectorizer_title_fc, tfidf_title_fc ,vectorizer_description ,tfidf_description,vectorizer_review ,tfidf_review= load_models()
# clf_lr,vectorizer_title,le1,le4,le7,le9,norm = load_models2()
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

def search(query, vectorizer):
  processed = re.sub("[^a-zA-Z0-9 ]", "", query.lower())
  query_vec = vectorizer.transform([processed])
  similarity = cosine_similarity(query_vec, tfidf_description).flatten()
  indices = np.argpartition(similarity, -50)[-50:]
  results = df_books_processed.iloc[indices]
  return results[['book_id', 'title_without_series', 'book_average_rating', 'book_url', 'cover_page']].head(5).style.format({'book_url': make_clickable, 'cover_page': show_image})
#
## similare reviwes


# Définition des fonctions de formatage
def make_clickable(val):
    return f'<a href="{val}" target="_blank">{val}</a>'

def show_image(val):
    return f'<img src="{val}" width="60">'



def review_similarity(book_id):
    # Entrer le book_id
    book_id=book_id.strip()
    # Ensure 'book_id' is of the same type in both DataFrames
    df_final_review['book_id'] = df_final_review['book_id'].astype(str)
    df_books_processed['book_id'] = df_books_processed['book_id'].astype(str)

    # Vérifier si le book_id existe dans df_final_review
    if book_id not in df_final_review['book_id'].astype(str).values:
        print(f"Error: book_id {book_id} not found in df_final_review.")
        print("Available book_ids in df_final_review:")
        print(df_final_review['book_id'].unique())
        return

    # Trouver l'index du livre
    index = df_final_review[df_final_review['book_id'] == book_id].index

    if len(index) == 0:
        print(f"Error: No reviews found for book_id {book_id}.")
        return

    # Calculer la similarité cosinus
    similarity = cosine_similarity(tfidf_review[index], tfidf_review).flatten()

    # Vérifier si la similarité a renvoyé des résultats
    if similarity.size == 0:
        print("Error: Similarity calculation failed. No data found.")
        return

    # Trouver les indices des livres les plus similaires
    indices = np.argpartition(similarity, -50)[-50:]
    book_ids = set(df_final_review.iloc[indices]['book_id'])

    # Créer un score pour chaque livre similaire
    score = [(score, book) for score, book in enumerate(book_ids)]
    df_score = pd.DataFrame(score, columns=['score', 'book_id'])

    # Ensure 'book_id' is consistent before merging
    book_ids = set(df_final_review.iloc[indices]['book_id'].astype(str))
    df_score = pd.DataFrame(score, columns=['score', 'book_id']).astype({'book_id': 'str'})

    # Obtenir les informations des livres similaires
    results = (df_books_processed[df_books_processed['book_id'].astype(str).isin(book_ids)]
            .merge(df_score, on='book_id')
            .sort_values(by='score'))

    # Afficher le titre du livre entré
    title_series = df_books_processed[df_books_processed['book_id'].astype(str) == book_id]['title_without_series']
    if not title_series.empty:
        book_title = title_series.values[0].strip()
        print('Entered book title: ', book_title)
    else:
        print('Error: Title not found for the entered book_id.')
        print("Available book_ids in df_books_processed:")
        print(df_books_processed['book_id'].unique())

    # Retourner les résultats
    return results[['book_id', 'title_without_series', 'book_average_rating', 'book_url', 'cover_page']].head(5).style.format({'book_url': make_clickable, 'cover_page': show_image})


# df_books_processed['book_id'] = df_books_processed['book_id'].astype(np.int64)
# df_final_review['book_id'] = pd.to_numeric(df_final_review['book_id'], errors='coerce')


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
    df_books_processed['read_at'] = pd.to_datetime(df_books_processed['read_at'], errors='coerce')

    # Remove timezone information to make them all offset-naive
    df_books_processed['read_at'] = df_books_processed['read_at'].apply(lambda x: x.tz_localize(None) if x is not pd.NaT and x.tzinfo else x)


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

#####################
from scipy.sparse import hstack

# Define the function to recommend books
def content_recommendation():
    user_id = st.text_input("Enter user_id:")
    if not user_id:
        st.warning("Please enter a user_id to get recommendations.")
        return

    # Fetch books not read by the user
    book_id = set(df_books_processed[df_books_processed['user_id'] == user_id]['book_id'])
    user_books = df_books_processed[~df_books_processed['book_id'].isin(list(book_id))].merge(df_final_review, on='book_id')
    user_books['user_id'] = len(user_books) * [user_id]
    user_books.reset_index(drop=True, inplace=True)
    user_books = user_books[user_books['book_id'].isin(le1.classes_)]
    user_books['book_id_mapped'] = le1.transform(user_books['book_id'])
    user_books['publisher_mapped'] = le4.transform(user_books['publisher'])
    user_books['is_ebook_mapped'] = le7.transform(user_books['is_ebook'])
    user_books['user_id_mapped'] = le9.transform(user_books['user_id'])

    # Transform title and review columns using vectorizers
    tfidf_title = vectorizer_title.transform(user_books['mod_title'])
    tfidf_review = vectorizer_review.transform(user_books['combined_processed_review'])

    # Create numeric features and scale them
    user_book_numeric = user_books[['book_id_mapped', 'publisher_mapped', 'is_ebook_mapped', 'user_id_mapped', 'publication_year', 'ratings_count', 'book_average_rating', 'num_pages']]
    data_scaled = norm.transform(user_book_numeric)
    data_scaled = hstack((data_scaled, tfidf_title, tfidf_review), dtype=np.float32)

    # Predict ratings
    prediction = clf_lr.predict(data_scaled.tocsr())
    user_books['rating'] = prediction

    # Top 50 recommendations
    top_50_books_for_user_content = user_books.sort_values(by=['rating'], ascending=False)[:50]
    book_title_liked_by_user = set(df_books_processed[df_books_processed['book_id'].isin(book_id)].sort_values(by='rating', ascending=False)['title_without_series'])

    # Display books highly rated by the user
    st.subheader("Books highly rated by given user:")
    for count, books in enumerate(list(book_title_liked_by_user)[:20]):
        st.write(f"{count + 1}. {books}")

    # Display top 5 recommended books with clickable URLs and cover images
    st.subheader("Top 5 Recommended Books:")
    top_5_books = top_50_books_for_user_content[['book_id', 'title_without_series', 'book_average_rating', 'book_url', 'cover_page']].head(5)
    st.dataframe(top_5_books.style.format({'book_url': make_clickable, 'cover_page': show_image}))



# Sidebar menu
st.sidebar.title("📚 Book Application")
menu_option = st.sidebar.radio("Select a function:", ["Autocomplete Book Search", "Top Concise Books", "Find Similar Books by Title", "Books Where User Choice's Match","Books having similar description","Books having similar review","Books vedettes,","Content Based Filtering"])

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

    # Input box for user ID
    query = st.text_input("🔍 Enter Book description:", "")
    if query:
       
        # Generate popular recommendations
        popular_recs = search(query,vectorizer_description)
        st.markdown('<table class="result-table">' + popular_recs.to_html(escape=False, index=False) + '</table>', unsafe_allow_html=True)
    else:
        st.warning("No similar books found.")
    
elif menu_option == "Books having similar review" :
    st.title("📚 Books having similar review")
    st.markdown("Enter a Book ID to get recommendations based on Books similar review.")

    # Input box for user ID
    query = st.text_input("🔍 Enter Book ID:", "")
    if query:
       
        # Generate popular recommendations
        popular_recs = review_similarity(query)
        st.markdown('<table class="result-table">' + popular_recs.to_html(escape=False, index=False) + '</table>', unsafe_allow_html=True)
    else:
        st.warning("No similar books found.")

# Adding to the sidebar menu
elif menu_option == "Books vedettes":
    books_vedette()


elif menu_option == "Content Based Filtering":
    st.title("📚 Content Based Filtering")

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