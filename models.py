

import pandas as pd
import pickle as pk
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# 

dataset = pd.read_csv('models/dataset_eda.csv.gzip', compression='gzip')
dataset['manufacturer'].fillna(value='', inplace=True)
users = set(dataset.reviews_username.values)

# Read model from pickle file

model_file_name = 'models/logistic_regression.pkl'
sentiment_model = pk.load(open(model_file_name, 'rb'))


# Read final user predicted ratings 

user_predicted_ratings_final = pd.read_csv('models/user_predicted_ratings_final.csv.gzip', compression='gzip')
user_predicted_ratings_final.set_index('reviews_username', inplace=True)


# Read preprocessed user sentiments

data_sentiment_preprocessed = pd.read_csv('models/data_sentiment_preprocessed.csv.gzip', compression='gzip')
data_sentiment_preprocessed.set_index('id', inplace=True)
data_sentiment_preprocessed = data_sentiment_preprocessed[data_sentiment_preprocessed['reviews_text_processed'].isnull() == False]


# Read TF-IDF vectorizer

tf_idf_vectorizer = pk.load(open('models/tf-idf-word_vectorizer.pkl', 'rb'))


def get_recommendations_filtered(username):

    recommendations = {'error_message': 'User not present in system.'}

    if(username in users):

        # Get top 20 recommendations for given username
        top_20_recommendations = user_predicted_ratings_final.loc[username].sort_values(ascending=False)[0:20]
        top_20_recommendations = pd.DataFrame(top_20_recommendations.index.values, columns=['Product ID'])

        top_20_recommendations_review = data_sentiment_preprocessed[data_sentiment_preprocessed.index.isin(top_20_recommendations['Product ID'].values)]
        
        # Get review for predicting sentiment
        X_test = top_20_recommendations_review['reviews_text_processed']
        # Transform text to vector data
        X_test_vct = tf_idf_vectorizer.fit_transform(X_test)

        # Predict sentiments for given comments
        y_test = sentiment_model.predict(X_test_vct)

        # Add data to results
        top_20_recommendations_review['sentiment'] = y_test

        # Get top 5 recommendations
        top_5_recommendations = top_20_recommendations_review.groupby(by=['id']).mean().sort_values(by="sentiment", ascending=False)[:5]


        user_data = dataset[dataset.id.isin(top_5_recommendations.index)][['id', 'name', 'brand', 'categories', 'manufacturer']].drop_duplicates()
        user_data.set_index('id', inplace=True)
        recommendations = user_data.T.to_json()

    return recommendations
