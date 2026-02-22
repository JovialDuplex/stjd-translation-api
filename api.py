from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import AutoTokenizer
import pycld2

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
@api.post("/two-ways-translation")
def translation_two_ways(text: str=Form(...)):
    reference_lang_inputs = {
        "en": 0,
        "fr": 1
    }

    reference_lang_outputs = {
        0: "en",
        1: "fr"
    }

    tokenizer_dict = {
        "fr": fr_en_tokenizer,
        "en": en_fr_tokenizer
    }
    model_dict = {
        "fr" : fr_en_model,
        "en": en_fr_model
    }
    
    #language detection 
    lang_detect_result = pycld2.detect(text)
    
    if lang_detect_result[0] == False:
        return {
            "message" : "translation impossible, an error has occured when detecting language "
        }
    
    lang_code = lang_detect_result[2][0][1]
    print("detected language : ", lang_code)

    #inputs of model 
    inputs = tokenizer_dict[lang_code](text, return_tensors="pt")
    generation_ids = model_dict[lang_code].generate(**inputs)

    result = tokenizer_dict[lang_code].decode(generation_ids[0], skip_special_tokens=True)
    
    return {
        "lang-source": lang_code,
        "lang-dest": reference_lang_outputs[not reference_lang_inputs[lang_code]],
        "text-source": text,
        "text-translate" : result
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