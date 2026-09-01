# LSTM ile Türkçe Metin Üretimi (Text Generation)

Bu proje, Türkçe günlük konuşma cümlelerini kullanarak sonraki kelimeyi tahmin eden ve verilen bir başlangıç ifadesine göre metin üreten Derin Öğrenme (LSTM) tabanlı bir NLP uygulamasıdır.

## 🚀 Teknolojiler
- **Python 3.10**
- **TensorFlow / Keras**
- **NumPy**
- **Docker**

## 📂 Proje Yapısı
- `train_lstm.py`: Veri işleme, model eğitimi ve metin üretimi aşamalarını içerir.
- `requirements.txt`: Gerekli Python kütüphaneleri.
- `Dockerfile`: Projenin konteynerize edilmiş ortam konfigürasyonu.

## 🛠️ Yerel Çalıştırma (Local)
```bash
pip install -r requirements.txt
python train_lstm.py

