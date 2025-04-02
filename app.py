import pandas as pd
import pickle
from flask import Flask, request, jsonify

# Load dataset and similarity matrix
df = pd.read_csv("your_dataset.csv")
with open('similarity_matrix.pkl', 'rb') as f:
    similarity_matrix = pickle.load(f)

# Flask app
app = Flask(__name__)

@app.route("/recommend", methods=["GET"])
def recommend():
    title = request.args.get("title")
    if title not in df["title"].values:
        return jsonify({"error": "Title not found"}), 404
    
    idx = df.index[df["title"] == title].tolist()[0]
    similarity_scores = list(enumerate(similarity_matrix[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    top_similar = [df.iloc[i[0]]["title"] for i in similarity_scores[1:6]]
    
    return jsonify({"recommendations": top_similar})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
