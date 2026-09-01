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
# 1. VERİ SETİ Hazırlığı
# ==========================================
data = [
    "bugün hava top oynamak için çok güzel",
    "sabah erkenden kalkıp yürüyüş yaptım",
    "arkadaşımla kahve içmek için sözleştik",
    "akşam yemeğinde sıcak bir çorba içtik",
    "ders çalışmak için kütüphaneye gittim",
    "otobüs durağında uzun süre beklemek zorunda kaldım",
    "yeni aldığım kitabı bir solukta bitirdim",
    "hafta sonu ailemle piknik yapmaya gideceğiz",
    "bilgisayarda kod yazarken zamanın nasıl geçtiğini anlamıyorum",
    "müzik dinleyerek dinlenmeyi çok seviyorum",
    "marketten ekmek ve süt alıp eve döndüm",
    "yağmur yağmaya başlayınca hemen şemsiyemi açtım",
    "telefonumun şarjı bitince kimseye ulaşamadım",
    "akşam arkadaşlarımla halı saha maçı yapacağız",
    "annem akşam yemeği için harika bir pasta yaptı",
    "otobüs kaçınca okula yürüyerek gitmek zorunda kaldım",
    "bu sabah saat alarmı çalmadığı için geç kaldım",
    "yeni bir film izlemek için mısır patlattık",
    "deniz kenarında oturup gün batımını izlemek harikaydı",
    "pazar günü evde dinlenip dizi izlemeyi planlıyorum",
    "sınav sonuçları açıklanınca herkes çok heyecanlandı",
    "öğle yemeğinde lezzetli bir sandviç yedim",
    "balkonda oturup taze çay içmek harika bir duygu",
    "parkta koşan çocukların neşesi herkesi güldürdü",
    "trafik o kadar yoğundu ki saatlerce yolda kaldık",
    "kedi bahçede hızlıca ağaca tırmandı",
    "yeni ayakkabım ayağımı biraz sıktığı için rahat edemedim",
    "kütüphanede sessizce çalışırken zaman hızlı geçti",
    "akşam saatlerinde sokaklar oldukça sakin oluyordu",
    "arkadaşımın doğum günü partisi için hediye arıyorum",
    "spordan sonra soğuk bir su içmek çok iyi geldi",
    "masanın üzerindeki anahtarlarımı almayı unutmuşum",
    "hafta içi her gün düzenli olarak spor yapmaya çalışıyorum",
    "güneş batarken gökyüzünün rengi inanılmaz güzelleşti",
    "yaz tatilinde sahilde yürümek çok keyifliydi",
    "kışın kar yağarken pencereden dışarıyı izlemeyi severim",
    "yeni proje üzerinde çalışırken bir hata ile karşılaştım",
    "kahvaltıyı kaçırınca öğlene doğru çok acıktım",
    "arkadaşım bana komik bir hikaye anlattı ve çok güldük",
    "çiçekleri sulamayı unuttuğum için biraz solmuşlar",
    "sipariş ettiğim kargo bugün öğleden sonra teslim edildi",
    "akşam yürüyüşü yaparken eski bir dostumla karşılaştım",
    "bisiklet sürerken rüzgarın yüzüme vurması rahatlatıcıydı",
    "ödevimi bitirip hemen uyumak istiyorum",
    "yeni aldığım kulaklık ile müzik dinlemek harika",
    "sabah kahvesini içmeden güne tam başlayamıyorum",
    "bahçedeki ağaçlar baharın gelmesiyle çiçek açtı",
    "sokak hayvanlarına bir kap su koymayı ihmal etmiyorum",
    "uzun bir aradan sonra memlekete gitmek iyi geldi",
    "akşam televizyonda güzel bir belgesel izledik",
    "fotoğraf çekmek için tarihi sokaklarda turladık",
    "yemeğe fazla tuz katınca tadı biraz bozuldu",
    "yorgun olduğum için akşam erken yattım",
    "hafta sonu temizlik yapmak bütün günümü aldı",
    "yeni aldığım bilgisayar çok hızlı çalışıyor",
    "otobüsteki yaşlı teyzeye yer verdim",
    "sınava girmeden önce son kez notlarımı gözden geçirdim",
    "rüzgar o kadar sert esti ki kapılar çarptı",
    "akşam çayının yanında bisküvi yemek harika oluyor",
    "yazın sıcağında dondurma yemek en büyük eğlencemiz",
    "kulaklığımı evde unuttuğum için yolculuk sıkıcı geçti",
    "akşam yemeğinden sonra kısa bir yürüyüşe çıktık",
    "yeni projenin sunumunu başarıyla tamamladık",
    "pencereden giren taze hava odayı ferahlattı",
    "çoğu zaman akşamları kitap okuyarak vakit geçiririm",
    "pazardan taze meyve ve sebze satın aldık",
    "yeni bir dil öğrenmek gerçekten sabır gerektiriyor",
    "kardeşimle birlikte yapboz yapmaktan çok keyif aldık",
    "yağmurdan sonra toprak kokusu her yeri kapladı",
    "tatile gitmeden önce bavulumu özenle hazırladım",
    "sokaktaki çocuklar sevinçle top peşinde koşuyordu",
    "arkadaşımın daveti üzerine akşam yemeğine gittik",
    "yeni yıl kararları arasında daha çok kitap okumak var",
    "sahilde yürürken martılara ekmek attık",
    "evde yalnız kalınca en sevdiğim müzikleri dinlerim",
    "ders aralarında bahçeye çıkıp hava alıyoruz",
    "yeni tarif denedim ve yemek düşündüğümden güzel oldu",
    "sabah koşusu yapmak gün boyu zinde kalmamı sağlıyor",
    "gece gökyüzünde yıldızları izlemek çok rahatlatıcı",
    "telefonuma gelen güncelleme ile cihaz daha akıcı oldu",
    "hafta sonu arkadaşlarla kamp yapmaya karar verdik",
    "sokak lambaları hava kararınca sırayla yandı",
    "tren yolculuğu yaparken dışarıyı izlemek çok huzurluydu",
    "odamı düzenleyip gereksiz eşyaları kenara ayırdım",
    "lezzetli bir akşam yemeğinin ardından tatlı yedik",
    "sabah erken kalkmak başta zor gelse de alışılıyor",
    "güzel bir filmin ardından uzun uzun sohbet ettik",
    "bahçedeki çimler uzayınca onları kesmek gerekti",
    "yeni bir hobi edinmek zihnimi rahatlatıyor",
    "gece yağmurunun sesiyle uyumak çok kolay oldu",
    "sabahları ilk iş olarak odayı havalandırırım",
    "yoğun bir günün ardından evde olmak çok güzel",
    "otobüs durağındaki kalabalık saat ilerledikçe azaldı",
    "soğuk günlerde sıcacık bir bitki çayı içmek iyi gelir",
    "projenin kodlarını yazıp başarıyla test ettik",
    "yeni aldığım deftere ilk notlarımı yazmaya başladım",
    "günün yorgunluğunu üzerimden atmak için duş aldım",
    "akşam saatlerinde parkta yürümek sakinleştirici oluyor",
    "arkadaşlarımla ortak bir proje üzerinde çalışıyoruz",
    "güzel bir pazar sabahına uyanmak insanı mutlu ediyor"
]

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