import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.decomposition import TruncatedSVD

def load_data():
    """
    Load the IMDB Top 1000 dataset and select relevant columns.
    """
    file_path = "imdb_top_1000.csv"
    df = pd.read_csv(file_path)
    df = df[['Series_Title', 'Overview', 'Genre', 'Director', 'Star1']].dropna()
    df.rename(columns={'Series_Title': 'title', 'Overview': 'description'}, inplace=True)
    return df

def preprocess_data(df):
    """
    Combine relevant textual features to create a richer representation for recommendations.
    """
    df['combined_features'] = df['description'] + ' ' + df['Genre'] + ' ' + df['Director'] + ' ' + df['Star1']
    return df

def recommend_movies(user_query, top_n):
    """
    Given a user's text input, recommend the top N similar movies using a hybrid approach.
    """
    df = load_data()
    df = preprocess_data(df)
    
    # Vectorize descriptions using TF-IDF
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_features'])
    
    # Dimensionality reduction using Truncated SVD for improved similarity measurement
    svd = TruncatedSVD(n_components=100)
    reduced_matrix = svd.fit_transform(tfidf_matrix)
    
    # Vectorize user query
    query_vector = tfidf_vectorizer.transform([user_query])
    query_reduced = svd.transform(query_vector)  
    # Compute cosine similarity using a more refined method
    similarity_scores = linear_kernel(query_reduced, reduced_matrix).flatten()
    
    # Get top N matches
    top_indices = similarity_scores.argsort()[-top_n:][::-1]
    
    # Return recommendations with similarity scores
    recommendations = df.iloc[top_indices][['title']]
    scores = similarity_scores[top_indices]
    
    return list(zip(recommendations['title'], scores))

def main():
    user_query = input("Enter your movie preference description: ")
    top_n = int(input("Enter the number of recommendations you want: "))
    
    recommendations = recommend_movies(user_query, top_n)
    print("\nTop Recommendations:")
    for title, score in recommendations:
        print(f"{title}: {score:.4f}")

if __name__ == "__main__":
    main()
 # type: ignore