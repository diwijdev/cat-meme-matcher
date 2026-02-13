from fastapi import FastAPI
from pydantic import BaseModel
from .matcher import match_meme
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Payload(BaseModel):
    mouth_open: float
    eye_open: float
    mouth_w: float
    smile_up: float
    brow_raise: float | None = None
    base_brow: float | None = None

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/match")
def match(p: Payload):
    label, conf = match_meme(p.model_dump(), p.base_brow)
    return {"label": label, "conf": conf}
