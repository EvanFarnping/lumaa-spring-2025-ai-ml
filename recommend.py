import argparse as parse
import pandas as pd
import numpy as np
import pprint

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_FILE_LINK = "https://raw.githubusercontent.com/peetck/IMDB-Top1000-Movies/refs/heads/master/IMDB-Movie-Data.csv"

def load_data(path_or_url : str = DEFAULT_FILE_LINK, 
              sample_size: int = 500,
              seed: int = 50) -> pd.DataFrame:
    """
    Loads and samples movie data from CSV (local/URL). 
    Returns DataFrame.
    """

    try:
        df = pd.read_csv(path_or_url)
        if sample_size > 0:
            df = df.sample(min(sample_size, len(df)), 
                           random_state=seed).reset_index(drop=True)
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to load the data: {str(e)}")

def get_text_columns(df : pd.DataFrame) -> pd.Series:
    """
    Identifies text-based columns. 
    Filters out numerical/role-specific columns that won't help much with 
    the "TF-IDF + cosine similarity approach."
    """

    try:
        text_columns = [
            col for col in df.columns
            if isinstance(df[col].iloc[0], str) 
            and any(ch.isalpha() for ch in df[col].iloc[0])
        ]

        # Features to remove that don't provide much contextual info, or just don't want.
        sub_str_to_remove = ["direct", "prod", "art", "song", "act", "title"]
        text_columns = [item.lower() for item in text_columns 
                        if all(sub not in item.lower() for sub in sub_str_to_remove)]
        
        return pd.Series(text_columns)
    except Exception as e:
        raise RuntimeError(f"Failed to get text only columns: {str(e)}")

def build_vectors(text_columns: pd.Series) -> tuple:
    """
    Creates TF-IDF vectors from text columns using:
    - English stopwords removal
    - 1-2 word ngrams
    - 5000 max features
    - Ignore terms that appear in <2 docs
    - Discard terms that appear in 80%< of docs
    Returns vectorizer and feature matrix.
    """

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=5000,
        min_df=2,
        max_df=0.8
    )
    tfidf_matrix = vectorizer.fit_transform(text_columns)
    return vectorizer, tfidf_matrix

def clean_query(query: str) -> str:
    """
    Removes common filler phrases from the query.
    Using spacy and rake_nltk would be better, especially for
    larger datasets.
    Returns preprocessed query.
    """

    unwanted_phrases = [
        "i love", "i like", "i want", "i am looking for",
        "i need", "can you find", "i'm looking for",
        "i'd love", "i prefer", "i'm searching",
        "find me", "i enjoy", "i look", "i'd want"
    ]

    query = query.lower()
    
    for phrase in unwanted_phrases:
        query = query.replace(phrase, "").strip()

    return query

def compute_similarity(query: str, 
                      vectorizer: TfidfVectorizer, 
                      tfidf_matrix: np.ndarray) -> np.ndarray:
    """
    Calculates cosine similarity between query vector and movie vectors.
    Returns similarity scores.
    """

    cleaned_query = clean_query(query)
    
    if not cleaned_query.strip():
        raise ValueError("Query cannot be empty.")
    
    query_vector = vectorizer.transform([cleaned_query])
    if query_vector.nnz == 0:
        return np.zeros(tfidf_matrix.shape[0])
    
    return cosine_similarity(query_vector, tfidf_matrix).flatten()

def recommend(query: str, df: pd.DataFrame, top_n: int = 5) -> list:
    """
    Combine helper functions into main function.

    1. Combine text features
    2. Vectorize data
    3. Calculate similarities
    4. Return top N valid recommendations
    """

    try:
        text_columns = get_text_columns(df)
        if text_columns.empty:
            raise ValueError("No text columns found for processing.")
        
        df.columns = df.columns.str.lower()
        combined_text = df[text_columns].fillna('').astype(str).agg(' '.join, axis=1)

        vectorizer, tfidf_matrix = build_vectors(combined_text)

        similarities = compute_similarity(query, vectorizer, tfidf_matrix)

        top_indices = np.argsort(similarities)[-top_n:][::-1]
        recommendations = [
            (df.iloc[idx]['title'], float(similarities[idx]))
            for idx in top_indices
            if similarities[idx] > 0  
        ]

        return recommendations
    
    except Exception as e:
        raise RuntimeError(f"Failed to generate recommendations: {str(e)}")

if __name__ == "__main__":
    parser = parse.ArgumentParser(
        description="AI/Machine Learning Intern Challenge: Simple Content-Based Recommendation.")
    parser.add_argument(
        "query", type=str, help="Your search query.")
    parser.add_argument(
        "--top_n", type=int, default=5, help="Number of recommendations to return.")
    parser.add_argument(
        "--data_loc", type=str, default=DEFAULT_FILE_LINK, help="URL or path to the dataset.")
    
    args = parser.parse_args()
    
    try:
        df = load_data(args.data_loc)
        
        recommendations = recommend(args.query, df, args.top_n)
        
        print(f"\nTop {len(recommendations)} recommendations for '{args.query}':")
        recommendations_list = []
        for i, (title, score) in enumerate(recommendations, 1):
            recommendation = f"{i}. {title} (Score: {score:.4f})"
            recommendations_list.append(recommendation)
        pprint.pprint(recommendations_list)
            
    except Exception as e:
        print(f"\nError: {str(e)}")
        exit(1)

