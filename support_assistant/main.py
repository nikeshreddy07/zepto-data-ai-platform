from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from pathlib import Path
from .rag import ask, AnswerResponse

app = FastAPI(title="Zepto Support Assistant")

BASE_DIR = Path(__file__).parent

@app.get("/", response_class=HTMLResponse)
async def get_index():
    index_path = BASE_DIR / "index.html"
    return index_path.read_text(encoding="utf-8")

class AskRequest(BaseModel):
    query: str

@app.post("/ask", response_model=AnswerResponse)
def ask_endpoint(request: AskRequest):
    return ask(request.query)
