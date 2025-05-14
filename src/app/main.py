from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app import login, csv_upload, dataset, describe, columns, chart, relationship, chatbot
from app.auth import TokenData, create_access_token
app = FastAPI()

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(login.router)
app.include_router(csv_upload.router)
app.include_router(dataset.router)
app.include_router(describe.router)
app.include_router(columns.router)
app.include_router(chart.router)
app.include_router(relationship.router)
app.include_router(chatbot.router)
