from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import AutoTokenizer

api = FastAPI()
api.add_middleware(CORSMiddleware, 
                   allow_credentials=True, 
                   allow_origins = ["*"],
                   allow_methods=["*"],
                   allow_headers=["*"])

# loading fr-en model and tokenizer 
fr_en_model_path ="marian-mt-fr-en-onnx"
fr_en_tokenizer = AutoTokenizer.from_pretrained(fr_en_model_path)
fr_en_model = ORTModelForSeq2SeqLM.from_pretrained(fr_en_model_path, use_cache=False)

#loading en-fr model an tokenizer 
en_fr_model_path = "marian-mt-en-fr-onnx"
en_fr_tokenizer = AutoTokenizer.from_pretrained(en_fr_model_path)
en_fr_model = ORTModelForSeq2SeqLM.from_pretrained(en_fr_model_path, use_cache=False)

print("loading model okay !")


@api.get("/")
def info():
    return {
        "API_Name": "translation fr->en and en->fr",
        "description": "this API can translate a french word to english and english words to french with a language detection "
    }

@api.post("/translation-en-fr")
def translation_en_fr(text: str= Form(...)):
    inputs = en_fr_tokenizer(text, return_tensors="pt")
    generation_ids = en_fr_model.generate(**inputs)
    result = en_fr_tokenizer.decode(generation_ids[0], skip_special_tokens=True)
    
    return {
        "text-source": text,
        "translation-text": result
    }

@api.post("/translation-fr-en")
def translation(text:str= Form(...)):
    inputs = fr_en_tokenizer(text, return_tensors="pt")
    generation_ids = fr_en_model.generate(**inputs)
    result = fr_en_tokenizer.decode(generation_ids[0], skip_special_tokens=True)
    
    return {
        "text-source": text,
        "translation-text": result
    }