
import os
import numpy as np
import faiss
from datetime import datetime
from supabase import create_client, Client
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration
from fastapi import FastAPI
from pydantic import BaseModel
from uvicorn import uvicorn
# === CONFIGURATION ===
SUPABASE_URL      = "https://qfobukmaxnrsedjnuoit.supabase.co"      # REPLACE with actual URL
SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFmb2J1a21heG5yc2Vkam51b2l0Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDg0MTk5NTcsImV4cCI6MjA2Mzk5NTk1N30.iHp9c_c3GnC4-TwgfJhqEgDzFxP9Nvy7IfLCI9NSvWA"                    # REPLACE with actual ANON KEY

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# === 1. Fetch articles from Supabase ===
def fetch_articles(limit=100):
    resp = supabase.table("news_items") \
                   .select("id, title, content") \
                   .limit(limit) \
                   .execute()
    return resp.data

# === 2. Summarization Pipeline (BART) ===
print("🔄 Loading summarization model...")
summ_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
summ_model     = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
summarizer     = pipeline("summarization", model=summ_model, tokenizer=summ_tokenizer)

def generate_and_store_summaries(articles):
    for art in articles:
        text = art["content"] or art["title"]
        summary = summarizer(text, max_length=150, min_length=40, do_sample=False)[0]["summary_text"]
        supabase.table("news_items") \
                .update({"summary": summary, "updated_at": datetime.utcnow().isoformat()}) \
                .eq("id", art["id"]) \
                .execute()

# === 3. Embedding & FAISS Index ===
print("🔄 Creating embeddings & FAISS index...")
articles = fetch_articles(limit=200)
texts    = [a["content"] or a["title"] for a in articles]
ids      = [a["id"] for a in articles]

embedder   = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode(texts, show_progress_bar=True)

dim   = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings, dtype='float32'))

# === 4. Chatbot using T5 Model (Better RAG Behavior) ===
print("🔄 Loading T5 chatbot model...")
t5_tokenizer = T5Tokenizer.from_pretrained("t5-base")
t5_model     = T5ForConditionalGeneration.from_pretrained("t5-base")

def chatbot_query(user_query, top_k=3):
    q_emb = embedder.encode([user_query])
    _, indices = index.search(np.array(q_emb, dtype='float32'), top_k)
    context = "\n\n".join([texts[i] for i in indices[0]])
    prompt = f"answer the question based on the context:\n{context}\n\nQuestion: {user_query}"
    inputs = t5_tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = t5_model.generate(inputs, max_length=150, num_beams=4, early_stopping=True)
    return t5_tokenizer.decode(outputs[0], skip_special_tokens=True)


# Import everything from your existing script file, e.g. main.py or just copy code here

# For example, if your script is called main.py:
# from main import chatbot_query, fetch_articles, generate_and_store_summaries

# But since you said no code change, just paste your code above here,
# or if in the same file, no import needed.

app = FastAPI()

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    user_message = request.message
    # Call your chatbot function to get answer
    reply = chatbot_query(user_message)
    return {"reply": reply}
if __name__ == "__main__":
    uvicorn.run("api_server:app", host="127.0.0.1", port=8000, reload=True)