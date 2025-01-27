from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from transformers import GPTNeoForCausalLM, GPT2Tokenizer

app = FastAPI()

# Loading the GPT-Neo model and tokenizer
model_name = "EleutherAI/gpt-neo-125M"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPTNeoForCausalLM.from_pretrained(model_name)

@app.on_event("startup")
def load_data():
    global entry_level, mid_level, senior_level, all_data
    try:
        # Loading datasets
        entry_level = pd.read_csv("data/entry_level.csv")
        mid_level = pd.read_csv("mid_level.csv")
        senior_level = pd.read_csv("senior_level.csv")
        all_data = pd.read_csv("Team_1.csv")  
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Dataset files not found.")

class FilterRequest(BaseModel):
    experience_level: str

class QueryRequest(BaseModel):
    query: str

# AI response generator
def get_ai_insights(query: str):
    """Function to generate response using GPT-Neo."""
    try:
        # Encode the input text
        inputs = tokenizer.encode(f"Analyze the following job market data and answer the question: {query}", return_tensors="pt")
        
        # AI response generator
        outputs = model.generate(inputs, max_length=200, num_return_sequences=1, no_repeat_ngram_size=2)
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error with GPT-Neo request: {e}")

# API Endpoints
@app.get("/data/preview")
def preview_data():
    # Preview of the dataset."""
    return entry_level.head(10).to_dict(orient="records")

@app.post("/data/filter")
def filter_data(filter_request: FilterRequest):
    # Filter data based on experience level
    experience_level = filter_request.experience_level.lower()
    
    if experience_level == "all":
        filtered_jobs = all_data
    elif experience_level == "entry-level":
        filtered_jobs = entry_level
    elif experience_level == "mid-level":
        filtered_jobs = mid_level
    elif experience_level == "senior-level":
        filtered_jobs = senior_level
    else:
        raise HTTPException(status_code=400, detail="Invalid experience level.")
    
    if filtered_jobs.empty:
        return {"message": f"No jobs found for the selected experience level: {filter_request.experience_level}."}

    return filtered_jobs.to_dict(orient="records")

@app.post("/ai/query")
def ai_query(query_request: QueryRequest):
    # Handle user queries 
    query = query_request.query
    
    # Get AI insights
    insights = get_ai_insights(query)
    return {"query": query, "insights": insights}

@app.get("/")
def root():
    # Health check 
    return {"message": "FastAPI backend is running!"}
