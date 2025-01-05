from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from backend.ner_model import extract_entities
import ast


app = FastAPI()

app.mount("/static", StaticFiles(directory="frontend"), name="static")

class TextInput(BaseModel):
    text: str

@app.post("/process_text")
async def process_text(text_input: TextInput):
    entities = extract_entities(text_input.text)
    processed_entities = process_entities(entities, text_input.text)
    return JSONResponse(content={"entities": processed_entities})

def process_entities(entities: list, input_text: str) -> list:
    text = input_text.split()
    entities_info = []
    processed_entities = []
    for i in range(len(entities)):
        if isinstance(entities[i], float):
            entities_info.append(float(entities[i]))
        else:
            entities_info.append(str(entities[i]))
        
        info_dict = ast.literal_eval(entities_info[i])
        index_value = info_dict.get('index', -1)
        
        if index_value != -1:
            index_value -= 1
        
        if 0 <= index_value < len(text):
            disease_word = text[index_value]
        else:
            disease_word = None
        
        processed_entity = {
            "entity_group": entities[i]["entity"],
            "score": float(entities[i]["score"]),
            "Disease": disease_word,
            "start": entities[i]["start"],
            "end": entities[i]["end"]
        }
        processed_entities.append(processed_entity)
    return processed_entities


@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    with open("frontend/index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)
