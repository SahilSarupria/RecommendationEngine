import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the reels data
reels_df = pd.read_csv("reels.csv")

# Load user ratings data
ratings_df = pd.read_csv("user_ratings.csv")

# Create a Reader object
reader = Reader(rating_scale=(1, 5))

# Load ratings into Surprise dataset format
data = Dataset.load_from_df(ratings_df[['user_id', 'reel_id', 'rating']], reader)

# Split the dataset into training and testing sets
trainset, testset = train_test_split(data, test_size=0.2)

# Build a collaborative filtering model (SVD)
algo = SVD()
algo.fit(trainset)

# Test the collaborative filtering model
predictions = algo.test(testset)

# Calculate and print accuracy metrics (RMSE, MAE)
print("Collaborative Filtering Metrics:")
print("RMSE:", accuracy.rmse(predictions))
print("MAE:", accuracy.mae(predictions))

# Content-based filtering using video features
# Using TF-IDF Vectorizer on the genres and descriptions
tfidf = TfidfVectorizer(stop_words='english')
reels_df['combined_features'] = reels_df['genre'] + ' ' + reels_df['description']
tfidf_matrix = tfidf.fit_transform(reels_df['combined_features'])

# Calculate cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to get content-based recommendations based on the reel title
def get_content_based_recommendations(reel_title, cosine_sim=cosine_sim, n_recommendations=5):
    # Get the index of the reel that matches the title
    idx = reels_df.index[reels_df['title'] == reel_title].tolist()[0]

    # Get the pairwise similarity scores of all reels with that reel
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the reels based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the top N most similar reels (excluding the first reel)
    sim_scores = sim_scores[1:n_recommendations + 1]  # Exclude the first reel (itself)

    # Get the reel indices
    reel_indices = [i[0] for i in sim_scores]

    # Return the top N most similar reels
    return reels_df[['title', 'reel_id']].iloc[reel_indices]

# Hybrid Recommendation Function
def hybrid_recommendation(user_id, n_recommendations=5):
    # Get all reels rated by the user
    user_rated_reels = ratings_df[ratings_df['user_id'] == user_id]['reel_id'].tolist()
    
    if not user_rated_reels:
        print(f"No ratings found for user {user_id}.")
        return []

    # Get content-based recommendations for each reel rated by the user
    content_based_recommendations = pd.DataFrame(columns=['title', 'reel_id'])
    
    for reel_id in user_rated_reels:
        reel_title = reels_df[reels_df['reel_id'] == reel_id]['title'].values[0]
        recommendations = get_content_based_recommendations(reel_title, n_recommendations=n_recommendations)
        content_based_recommendations = pd.concat([content_based_recommendations, recommendations]).drop_duplicates()

    # Remove already rated reels
    content_based_recommendations = content_based_recommendations[~content_based_recommendations['reel_id'].isin(user_rated_reels)]
    
    # Return unique recommended reels
    return content_based_recommendations.drop_duplicates().head(n_recommendations)

# Example of hybrid recommendation
user_id = 1
recommended_reels = hybrid_recommendation(user_id)

# Print recommended reels for the user
print(f"\nRecommended Reels for User {user_id}:")
print(recommended_reels[['title', 'reel_id']])
