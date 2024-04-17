from flask import Flask, render_template, request
import pickle
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json

app = Flask(__name__)

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)


model = load_model('model.h5')
df = pd.read_csv("Tweets.csv")
tweet_df = df[['text','airline_sentiment']]
tweet_df = tweet_df[tweet_df['airline_sentiment'] != 'neutral']
sentiment_label = tweet_df.airline_sentiment.factorize()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        tw = tokenizer.texts_to_sequences([text])
        tw = pad_sequences(tw, maxlen=200)
        prediction = int(model.predict(tw).round().item())
        sentiment = sentiment_label[1][prediction]
        return render_template('result.html', text=text, sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)
