# Import necessary libraries
import streamlit as st
import pandas as pd
import re
from elasticsearch import Elasticsearch

# Initialize Elasticsearch connection
es = Elasticsearch("https://2eb9-102-97-167-182.ngrok-free.app", timeout=60, max_retries=10, retry_on_timeout=True)

# CSS styling for images and zoom effect
st.markdown("""
    <style>
    .cover-img {
        width: 60px;
        height: auto;
        transition: transform 0.3s ease;  /* Smooth zoom transition */
        cursor: pointer;
    }
    .cover-img.zoomed {
        transform: scale(2);  /* Zoom level */
        transition: transform 0.3s ease;
    }
    </style>
""", unsafe_allow_html=True)

# JavaScript for image click-to-zoom
st.markdown("""
    <script>
    // Function to toggle the zoom on images
    function toggleZoom(event) {
        // Get the clicked image
        const img = event.target;
        // Toggle the "zoomed" class
        img.classList.toggle("zoomed");
    }

    // Attach click event listener to all images with class "cover-img"
    document.addEventListener("DOMContentLoaded", function() {
        const images = document.querySelectorAll('.cover-img');
        images.forEach(img => {
            img.addEventListener('click', toggleZoom);
        });
    });
    </script>
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
