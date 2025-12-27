import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from flask import Flask, request, jsonify
from flask_cors import CORS
import random
import time
import os
import requests


TWITTER_BEARER_TOKEN = "YOUR BEARER TOKEN GOES HERE" 

#Load Model and Dependencies
print("Loading dependencies")
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

port_stem = PorterStemmer()

try:
    model = pickle.load(open('trained_model.sav', 'rb')) 
    vectorizer = pickle.load(open('vectorizer.sav', 'rb'))
    print("Model and Vectorizer has been loaded successfully")
except FileNotFoundError:
    print("ERROR: 'trained_model.sav' or 'vectorizer.sav' not found.")
    model = None
    vectorizer = None
except Exception as e:
    print(f"An error occurred while loading the model: {e}")
    model = None
    vectorizer = None

#Define the SAME clean_text function
def clean_text(text):
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [port_stem.stem(w) for w in text if w not in stopwords.words('english') and len(w) > 1]
    return ' '.join(text)

#Tweet Fetching Functions

def fetch_real_tweets(hashtag):
    if TWITTER_BEARER_TOKEN == "YOUR BEARER TOKEN GOES HERE":
        print("WARNING: TWITTER BEARER TOKEN is not set. Using Dummy data instead.")
        print("Please get your Bearer Token ")
        return fetch_simulated_tweets(hashtag) # Fallback to simulation

    print(f"Calling API for hashtag: #{hashtag}")
    
    # This is the v2 endpoint for recent tweet search
    url = "https://api.twitter.com/2/tweets/search/recent"
    
    # Set the headers with your Bearer Token for authentication
    headers = {
        "Authorization": f"Bearer {TWITTER_BEARER_TOKEN}"
    }
    
    # Set the query parameters

    params = {
        'query': f"#{hashtag} -is:retweet lang:en",
        'max_results':50
    }
    
    try:
        response = requests.get(url, headers=headers, params=params)
        data = response.json()
        
        if response.status_code == 200:
            if 'data' in data:
                # this block of code will extract just the "text" from each tweet
                tweets = [tweet['text'] for tweet in data['data']]
                return tweets
            else:
                # this will show that there is no tweets found for that hashtag
                return [f"No recent tweets found for #{hashtag}."]
        else:
            print(f"Error from X API: {data.get('detail', 'Unknown error')}")
            return [f"Error: Could not fetch tweets. {data.get('detail', '')}"]
            
    except Exception as e:
        print(f"Error calling X API: {e}")
        return [f"Error: {e}"]

def fetch_simulated_tweets(hashtag):
    print(f"Simulating API call for: #{hashtag}")
    all_dummies = []
    try:
        with open('dummy_tweets.txt', 'r') as f:
            for line in f:
                if line.startswith("POS:") or line.startswith("NEG:"):
                    all_dummies.append(line.replace("#{hashtag}", f"#{hashtag}"))
                        
    except FileNotFoundError:
        print("ERROR: dummy_tweets.txt not found!")
        return ["Error: dummy_tweets.txt not found."]
    
    random.shuffle(all_dummies)
    # Return 100 dummies if we have them, otherwise just return all
    return all_dummies[:100]

#Create Flask App and API Endpoint
app = Flask(__name__)
CORS(app)

@app.route('/predict-hashtag', methods=['POST'])
def predict_hashtag():
    if not model or not vectorizer:
        return jsonify({'error': 'Model or Vectorizer is not loaded on the server.'}), 500
        
    data = request.get_json()
    hashtag = data.get('hashtag', '').lstrip('#')
    
    if not hashtag:
        return jsonify({'error': 'No hashtag provided.'}), 400
    
    try:
       
        text_list = fetch_real_tweets(hashtag)
        
        
        results = []
        for text in text_list:
            cleaned_text = clean_text(text)
            vectorized_text = vectorizer.transform([cleaned_text])
            prediction_raw = model.predict(vectorized_text)
            prediction = int(prediction_raw[0]) 
            sentiment = 'Positive' if prediction == 1 else 'Negative'
            
            results.append({
                'tweet': text,
                'sentiment': sentiment,
                'cleaned_text': cleaned_text
            })
            
        # Step 3: Return the results as JSON
        return jsonify(results)

    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return jsonify({'error': str(e)}), 500

# Run the App 
if __name__ == '__main__':
    print("Starting Flask backend server on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)