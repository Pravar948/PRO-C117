import tensorflow as tf
import pandas as pd
from flask import Flask, render_template, request

app = Flask(_name_)

# load the model and the dataset
model = tf.keras.models.load_model('emotion_model.h5')
data = pd.read_csv('reviews.csv')

def predict_emotion(review):
    # preprocess the review before making a prediction
    review = review.lower()
    review = tf.keras.preprocessing.text.text_to_word_sequence(review)
    review = tf.keras.preprocessing.sequence.pad_sequences([review], maxlen=100)
    # make the prediction
    prediction = model.predict(review)
    emotion = data.emotion[prediction.argmax()]
    return emotion

@app.route('/')
def index():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    emotion = predict_emotion(review)
    return render_template('result.html', emotion=emotion)

if _name_ == '_main_':
    app.run(debug=True)