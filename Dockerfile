# 1. Taban imajı belirle
FROM python:3.10-slim

# 2. Çalışma dizinini ayarla
WORKDIR /app

# 3. Bağımlılık listesini konteynere kopyala ve yükle
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 4. Projedeki tüm dosyaları konteynere kopyala
COPY . .

# 5. Konteyner çalıştığında çalışacak varsayılan komutu belirt
CMD ["python", "train_lstm.py"]