import pandas as pd
import pickle
from flask import Flask, request, jsonify

# Load the dataset and similarity matrix
df = pd.read_csv("your_dataset.csv")
with open('similarity_matrix.pkl', 'rb') as f:
    similarity_matrix = pickle.load(f)

# Initialize the Flask app
app = Flask(__name__)

@app.route("/recommend", methods=["GET"])
def recommend():
    title = request.args.get("title")
    
    # If title is not found in dataset
    if title not in df["title"].values:
        return jsonify({"error": "Title not found"}), 404
    
    # Find the index of the given title
    idx = df.index[df["title"] == title].tolist()[0]
    
    # Get similarity scores and sort them
    similarity_scores = list(enumerate(similarity_matrix[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    # Get top 5 recommendations
    top_similar = [df.iloc[i[0]]["title"] for i in similarity_scores[1:6]]
    
    # Return recommendations as JSON
    return jsonify({"recommendations": top_similar})

# Vercel expects a special handler for serverless functions
def handler(req, res):
    return app(req, res)

# Ensure that we use the handler function for serverless
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
