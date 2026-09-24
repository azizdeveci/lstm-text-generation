from pydantic import BaseModel

class TextRequest(BaseModel):
    seed_text: str
    next_words: int
