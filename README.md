# 📚 Book Recommendation System

A comprehensive book recommendation system powered by machine learning and advanced search capabilities, featuring real-time autocomplete, multiple recommendation algorithms, and content-based filtering.

## 🌟 Features

### 🔍 Real-Time Autocomplete Search
- Instant book title suggestions as you type
- Powered by Elasticsearch with fuzzy matching
- Fast and responsive search experience

### 📖 Multiple Recommendation Approaches

1. **Content-Based Filtering**
   - Book recommendations based on titles, descriptions, and reviews
   - TF-IDF vectorization for text similarity
   - Machine learning models (Logistic Regression, Decision Tree, Random Forest)

2. **Collaborative Filtering**
   - User-User similarity recommendations
   - Item-Item similarity recommendations
   - SVD (Singular Value Decomposition) approach

3. **Top Books Recommendations**
   - Highest rated books
   - Concise books (under 300 pages)
   - Featured books by date
   - Popular books recommendations

4. **Similarity-Based Recommendations**
   - Books with similar titles
   - Books with similar descriptions
   - Books with similar reviews
   - Book cover analysis using CNN (VGG16)

5. **Special Features**
   - Correlation-based recommendations
   - Popular books matching user preferences
   - Time-based featured books

## 🛠️ Technologies Used

- **Python 3.x**
- **Web Framework**: Streamlit
- **Search Engine**: Elasticsearch
- **Machine Learning**:
  - Scikit-learn (Logistic Regression, Decision Tree, Random Forest)
  - Surprise Library (SVD)
  - TensorFlow/Keras (VGG16 for image analysis)
- **Data Processing**: Pandas, NumPy
- **Text Processing**: TF-IDF Vectorization, Cosine Similarity
- **Visualization**: Matplotlib, Seaborn

## 📁 Project Structure

```
.
├── app.py                          # Main Streamlit app with autocomplete
├── streamfront.py                  # Full Streamlit frontend with all features
├── system_rec_(1)_(2)_(3)_(3)_(3).py  # Complete ML implementation notebook
├── Systeme_d_auto_completion.ipynb    # Autocomplete system notebook
├── download.ipynb                  # Data download notebook
├── sys-rec/                        # Additional system files
│   ├── page2.py
│   ├── page3.py
│   ├── page4.py
│   ├── le4.pkl                     # Label encoder for publisher
│   ├── le7.pkl                     # Label encoder for is_ebook
│   └── norm.pkl                    # Normalization scaler
└── data/                           # Data directory
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install streamlit pandas numpy scikit-learn
pip install elasticsearch scikit-surprise
pip install tensorflow keras
pip install matplotlib seaborn
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/book-recommendation-system.git
cd book-recommendation-system
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up Elasticsearch:
   - Install and configure Elasticsearch
   - Update the Elasticsearch URL in `app.py` and `streamfront.py`
   - Index your book data

4. Prepare data:
   - Place your processed data files (`df_books_users_processed.csv.gz`, `df_final_review.csv.gz`) in the project directory
   - Ensure all pickle files (`.pkl`) are present

5. Run the application:
```bash
streamlit run streamfront.py
```

## 💡 Usage

### Autocomplete Book Search
- Start typing in the search box
- Get instant suggestions with book details, ratings, and cover images

### Book Recommendations
1. **Similar User Recommendations**: Enter a user ID to get books liked by similar users
2. **Similar Books**: Enter a book title to find similar books
3. **Description Search**: Enter book description keywords to find matching books
4. **Review Similarity**: Enter a book ID to find books with similar reviews
5. **Top Concise Books**: View highly-rated short books
6.残酷 **Featured Books**: Enter a date to see trending books since that date

## 📊 Data Sources

This project uses Goodreads book data including:
- Book titles, descriptions, and metadata
- User ratings and reviews
- Publication information
- Book cover images
- Reading statistics

## 🎯 Key Algorithms

### 1. Content-Based Filtering
- Uses TF-IDF vectorization on book titles, descriptions, and reviews
- Implements multiple ML classifiers for rating prediction
- Combines text features with numeric features (pages, ratings, etc.)

### 2. Collaborative Filtering
- **User-User**: Finds users with similar reading preferences
- **Item-Item**: Recommends books similar to ones you've liked
- **SVD**: Matrix factorization for rating prediction

### 3. Image-Based Recommendations
- Uses VGG16 CNN for book cover feature extraction
- Cosine similarity on image embeddings
- Visual similarity recommendations

### 4. Hybrid Approach
- Combines multiple recommendation methods
- Weighted scoring system
- Popularity-biased recommendations

## 📈 Model Performance

The system includes multiple ML models:
- **Logistic Regression**: Best overall performance for rating classification
- **Decision Tree**: Interpretable recommendations
- **Random Forest**: Robust ensemble method
- **SVD**: Collaborative filtering with RMSE optimization

## 🔧 Configuration

### Elasticsearch Setup
Update the Elasticsearch connection in the code:
```python
es = Elasticsearch("your-elasticsearch-url", timeout=60, max_retries=10, retry_on_timeout=True)
```

### Model Files
Ensure the following pickle files are present:
- `vectorizer_title.pkl`
- `vectorizer_description.pkl`
- `vectorizer_review.pkl`
- `tfidf_title.pkl`
- `tfidf_description.pkl`
- `tfidf_review.pkl`
- `clf_lr.pkl` (or other model files)
- `le1.pkl`, `le4.pkl`, `le7.pkl`, `le9.pkl` (label encoders)
- `norm.pkl` (normalization scaler)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is open source and available under the MIT License.

## 👤 Author

**Hamza**

## 🙏 Acknowledgments

- Goodreads for providing the book dataset
- Scikit-learn and TensorFlow communities
- Elasticsearch for powerful search capabilities

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

⭐ **Star this repo if you find it helpful!**
