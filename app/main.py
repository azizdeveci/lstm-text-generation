from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from app.routes import text_generation

# Swagger UI (/docs) için API metadatalarını zenginleştirdik
app = FastAPI(
    title="LSTM Text Generator API",
    description="Türkçe metin tahmini ve üretimi yapan LSTM modeli için API ve Arayüz servisi.",
    version="1.0.0"
)

# İleride front-end'i farklı bir porta veya React/React Native gibi ayrı bir yapıya taşırsan 
# tarayıcı tarafında CORS hatası almamak için Middleware ekliyoruz.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Prod ortamında spesifik domainleri yazabilirsin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Router'ı dahil et
app.include_router(text_generation.router)

# Templates ve static klasörlerini tanıt
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# FastAPI'nin doğasına uygun olarak endpoint'i asenkron (async) hale getirdik
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})