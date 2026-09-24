# 1. Taban imajı (hafif ve güvenli)
FROM python:3.10-slim

# 2. Çalışma dizini
WORKDIR /app

# 3. Sistem paketlerini güncelle ve bağımlılıkları yükle
COPY requirements.txt .
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        libhdf5-dev \
        && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# 4. Projedeki tüm dosyaları konteynere kopyala
COPY . .

# 5. Modeli konteynerin kendi kütüphaneleriyle sıfırdan eğit
RUN python train_lstm.py

# 6. FastAPI'nin çalışacağı portu dışarı aç
EXPOSE 8000

# 7. Konteyner çalıştığında FastAPI sunucusunu Uvicorn ile başlat
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
