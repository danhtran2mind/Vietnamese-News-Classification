import gradio as gr
import tensorflow as tf
import pickle
import json
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
from underthesea import word_tokenize
import numpy as np

# Load Model
model = tf.keras.models.load_model('saved_models/bidirectional-GRU.h5')

# Load Tokenizer
with open('tokenizers/tokenizer.pkl', 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer)

# Load the Label Dictionary
with open("saved_models/label_dict.json", 'r', encoding='utf-8') as file:
    label_dict = json.load(file)

MAX_LEN = 8054
characters_to_replace = r'!"#$%&()*+,-./:;=?@\[\\\]^`{|}~\t\n'

def predict_news_type(content):
    content = re.sub(f"[{re.escape(characters_to_replace)}]", "", content.replace("\n", " "))
    content = "<sos> " + word_tokenize(content.replace("\n", ""), format="text") + " <eos>"
    content = tokenizer.texts_to_sequences([content])
    content = pad_sequences(content, maxlen=MAX_LEN, padding='post')
    content_predict = model.predict(content, verbose=0)
    result = np.argmax(content_predict, axis=1)
    category = label_dict[str(result[0])]
    
    # Get all categories and their probabilities
    probabilities = content_predict[0].tolist()
    category_probabilities = {label_dict[str(i)]: prob for i, prob in enumerate(probabilities)}
    
    return category, category_probabilities

# Create Gradio Interface
demo = gr.Interface(
    fn=predict_news_type,
    inputs=gr.Textbox(label="Enter the news content"),
    outputs=[
        gr.Textbox(label="Predicted News Category"),
        gr.JSON(label="Category Probabilities")
    ],
    title="News Type Prediction",
    description="Enter the news content to predict its category and see the probabilities for all categories."
)
if __name__ == "__main__":
    # Launch the Gradio Interface
    demo.launch()
