# Import necessary libraries
import streamlit as st
import pandas as pd
import re
from elasticsearch import Elasticsearch

# Initialize Elasticsearch connection
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

# Function for auto-completion using Elasticsearch
def autocomplete_title(user_input):
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
            'Title': source['title_without_series'].title(),  # Capitalize title for display
            'Publication Year': source['publication_year'],
            'Publisher': source['publisher'],
            'Average Rating': source['book_average_rating'],
            'Cover Page': source['cover_page'],
            'Book URL': source['book_url']
        })
    
    # Convert data to DataFrame
    return pd.DataFrame(data)

# Streamlit app layout
st.title("📚 Real-Time Book Autocomplete System")
st.markdown("Start typing a book title to receive suggestions in real time! 🎉")

# Input box for real-time autocomplete
user_input = st.text_input("🔍 Search for a book:", "")

# Real-time search execution upon input change
if user_input:
    results = autocomplete_title(user_input)
    
    # Display the DataFrame if results exist
    if not results.empty:
        # Format Cover Page as an image and Book URL as a clickable link
        results['Cover Page'] = results['Cover Page'].apply(lambda url: f'<img src="{url}" class="cover-img">')
        results['Book URL'] = results['Book URL'].apply(lambda url: f'<a href="{url}" target="_blank" class="book-link">More Info</a>')
        
        # Render as HTML to display images and links
        st.markdown('<table class="result-table">' + results.to_html(escape=False, index=False) + '</table>', unsafe_allow_html=True)
    else:
        st.warning("No results found. Please try a different search term.")
else:
    st.info("Type in the box above to begin your search!")
