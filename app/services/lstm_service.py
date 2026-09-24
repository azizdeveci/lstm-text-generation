import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import os

# Model ve tokenizer yükleme
MODEL_PATH = "model/lstm_text_generator_model.keras"
model = load_model(MODEL_PATH)

# Tokenizer'ı yeniden eğitmek yerine kaydetmek daha iyi olurdu.
# Şimdilik sentences.txt üzerinden yeniden oluşturuyoruz.
from tensorflow.keras.preprocessing.text import Tokenizer

data = []
with open("data/sentences.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            data.append(line)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(data)
total_words = len(tokenizer.word_index) + 1
max_sequence_length = max(len(tokenizer.texts_to_sequences([t])[0]) for t in data)

def generate_text_with_model(seed_text: str, next_words: int):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_length-1, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        predicted_word_index = np.argmax(predicted, axis=-1)[0]

        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_word_index:
                output_word = word
                break

        seed_text += " " + output_word
    return seed_text
