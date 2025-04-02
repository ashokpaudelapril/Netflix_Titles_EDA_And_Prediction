import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

import kagglehub

# Download latest version
path = kagglehub.dataset_download("muhammadtahir194/netflix-movies-and-tv-shows-dataset")

print("Path to dataset files:", path)

import os

# List all files in the dataset directory to identify the dataset file
files = os.listdir(path)
print("Available files:", files)

# If the dataset contains a CSV file, read it
csv_file = [f for f in files if f.endswith('.csv')][0]  # Get the first CSV file

def process_data():
    # Load dataset
    df = pd.read_csv(os.path.join(path, csv_file))
    df = df.fillna("")

    # Create combined features for recommendation
    df["combined_features"] = df["title"] + " " + df["director"] + " " + df["cast"] + " " + df["listed_in"] + " " + df["description"]

    # TF-IDF Vectorization
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df["combined_features"])

    # Compute Cosine Similarity
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Save similarity matrix for app.py usage
    with open('similarity_matrix.pkl', 'wb') as f:
        pickle.dump(similarity_matrix, f)
    
    return df, similarity_matrix

if __name__ == "__main__":
    process_data()  # Execute the function to process the data and save the similarity matrix
