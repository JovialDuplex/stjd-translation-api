from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware

api = FastAPI()
api.add_middleware(CORSMiddleware, 
                   allow_credentials=True, 
                   allow_origins = ["*"],
                   allow_methods=["*"],
                   allow_headers=["*"])

@api.get("/")
def info():
    return {
        "API_Name": "translation fr->en and en->fr",
        "description": "this API can translate a french word to english and english words to french with a language detection "
    }