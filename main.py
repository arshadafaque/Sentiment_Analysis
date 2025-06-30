from fastapi import FastAPI
from pydantic import BaseModel
import pickle

# Load pickled files
with open("svc_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

app = FastAPI()

class InputText(BaseModel):
    text: str

@app.post("/predict")
def predict_sentiment(input: InputText):
    vec = vec = vectorizer.transform([input.text]).toarray()
    pred = model.predict(vec)[0]
    return {"sentiment": "Positive" if pred == 0 else "Negative"}
