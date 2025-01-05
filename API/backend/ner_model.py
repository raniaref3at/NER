from transformers import BertTokenizer, BertForTokenClassification, pipeline

model_path = "/Users/nono/API/model/biobert-fine-tuned-ner/checkpoint-3750"

tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForTokenClassification.from_pretrained(model_path)
nlp = pipeline("ner", model=model, tokenizer=tokenizer)

def extract_entities(text):
    results = nlp(text)
    entities = [
        {"entity": result['entity'], "score": result['score'], "index": result['index'], "start": result['start'], "end": result['end']}
        for result in results
    ]
    return entities
