from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from model import Item

app = FastAPI()

# CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post('/comment')
async def receive_data(item: Item):
    comment = 'Hello, world!'
    return {'comment': comment}