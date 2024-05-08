from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.define_model import define_model

app = FastAPI()

llm_chain = define_model()

class UserQuery(BaseModel):
    question: str

@app.post("/chat/")
def chat_with_llama(query: UserQuery):
    try:
        # Using the LangChain to run the query through the model
        answer = llm_chain.run(query.question)
        return {"answer": answer}
    except Exception as e:
        # Better error handling to diagnose issues effectively
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "LLAMA Chatbot is running. Use the /chat endpoint to interact."}
