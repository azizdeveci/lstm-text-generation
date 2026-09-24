from fastapi import APIRouter
from app.models.text_request import TextRequest
from app.services.lstm_service import generate_text_with_model

router = APIRouter(prefix="/generate", tags=["Text Generation"])

@router.post("/")
def generate_text(request: TextRequest):
    result = generate_text_with_model(request.seed_text, request.next_words)
    return {"input": request.seed_text, "output": result}
