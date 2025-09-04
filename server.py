# server.py
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import joblib


# ============ تحميل النموذج و الـ TF-IDF ============
linear_svc = joblib.load("news_classifier_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")
# print(linear_svc.classes_)
# ============ المابنج للفئات ============
class_mapping = {
    1: "World",
    2: "Sports",
    3: "Business",
    4: "Sci/Tech"
}

# ============ FastAPI App ============
app = FastAPI(title="News Classification API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # You can restrict this later e.g. ["http://127.0.0.1:5500"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# شكل الداتا اللي جايه في الـ request
class NewsInput(BaseModel):
    text: str

# ============ دالة التنبؤ ============
def preprocess_text(text):
    import re, nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    nltk.download('stopwords')
    nltk.download('punkt_tab')
    nltk.download('wordnet')
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)  # حذف الروابط
    text = re.sub(r"<.*?>", " ", text)  # حذف HTML
    text = re.sub(r"[^a-z\s]", " ", text)

    tokens = nltk.word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)

# Endpoint للتصنيف
@app.post("/predict")
def predict_category(input: NewsInput):
    try:
        clean_text = preprocess_text(input.text)
        text_tfidf = tfidf.transform([clean_text])
        pred = linear_svc.predict(text_tfidf)[0]
        print("Predicted index:", pred)  # Debug
        category = class_mapping.get(pred, "Unknown Category")
        return {"prediction": category}  # Return dictionary
    except Exception as e:
        return {"error": str(e), "prediction": "Unknown Category"}
    
if __name__ == "__main__":

    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)


