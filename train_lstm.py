"""
LSTM (Long Short-Term Memory) ile Doğal Dil İşleme ve Metin Üretimi

Bu proje, Gemini tarafından oluşturulan Türkçe günlük hayat cümlelerini kullanarak
kelime bazlı n-gram dizileri türetir ve sıradaki kelimeyi tahmin eden bir LSTM modeli eğitir.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

print("Kütüphaneler başarıyla yüklendi.")

# ==========================================
# 1. VERİ SETİ Hazırlığı (sentences.txt'den okuma)
# ==========================================
data = []
with open("data/sentences.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:  # boş satırları atla
            data.append(line)

print(f"Toplam {len(data)} cümle yüklendi.")


# ==========================================
# 2. TOKENIZATION (Metni Sayısal Temsillere Dönüştürme)
# ==========================================
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data)
total_words = len(tokenizer.word_index) + 1  # Index 0 padding (doldurma) için ayrılır
print(f"Toplam benzersiz kelime sayısı: {total_words}")

# ==========================================
# 3. N-GRAM DİZİLERİ OLUŞTURMA
# ==========================================
# Her cümleden sıralı kelime kombinasyonları (n-gram dizileri) üretilir
input_sequences = []
for text in data:
    token_list = tokenizer.texts_to_sequences([text])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# ==========================================
# 4. PADDING (Dizi Boyutlarını Eşitleme)
# ==========================================
max_sequence_length = max(len(x) for x in input_sequences)
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre'))

# ==========================================
# 5. GİRDİ (X) VE HEDEF (y) DEĞİŞKENLERİNİ AYIRMA
# ==========================================
X = input_sequences[:, :-1]  # Son kelimeye kadar olan girdi dizisi
y = input_sequences[:, -1]   # Tahmin edilecek hedef kelime

# Hedef kelimeleri kategorik One-Hot vektörlere dönüştür
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

# ==========================================
# 6. LSTM MODEL MİMARİSİ
# ==========================================
model = Sequential([
    Embedding(total_words, 50, input_length=max_sequence_length-1),
    LSTM(100),
    Dense(total_words, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ==========================================
# 7. MODEL EĞİTİMİ
# ==========================================
print("\nModel eğitimi başlatılıyor...")
model.fit(X, y, epochs=100, verbose=1)
model.save("model/lstm_text_generator_model.keras")
# ==========================================
# 8. METİN ÜRETİM (TEXT GENERATION) FONKSİYONU
# ==========================================
def generate_text(seed_text, next_words, model, max_sequence_length):
    """
    Verilen başlangıç kelimesinden/cümlesinden itibaren belirlenen sayı kadar
    ardışık kelime tahmini yapar.
    """
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

# ==========================================
# 9. MODEL TESTİ
# ==========================================
seed = "bugün hava"
generated_output = generate_text(seed, 5, model, max_sequence_length)
print("\n--- ÜRETİLEN METİN ---")
print(f"Girdi: '{seed}'")
print(f"Çıktı: '{generated_output}'")