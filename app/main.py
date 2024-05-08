from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.define_model_llamacpp import define_model_llm_chain
from app.define_model_ctransformers import define_model_ctransformers
from app.define_model_llama_cpp_python import define_model_llama_cpp_python


app = FastAPI()

llm_chain = define_model_llm_chain()
llm = define_model_ctransformers()
llm_Llama = define_model_llama_cpp_python()

class UserQuery(BaseModel):
    question: str

@app.post("/chat-llamacpp/")
def chat_with_llama(query: UserQuery):
    try:
        # Using the LangChain to run the query through the model
        answer = llm_chain.run(query.question)
        return {"answer": answer}
    except Exception as e:
        # Better error handling to diagnose issues effectively
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat-ctransformers/")
def chat_with_ctransformers(query: UserQuery):
    try:
        answer = llm(query.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat-llamacpp-python/")
def chat_with_llama_python(query: UserQuery):
    try:
        answer = llm_Llama(
            f"Q: {query.question} A: ", # Prompt
            max_tokens=32, # Generate up to 32 tokens, set to None to generate up to the end of the context window
            stop=["Q:", "\n"], # Stop generating just before the model would generate a new question
            echo=True # Echo the prompt back in the output
        ) # Generate a completion, can also call create_completion
        return {"answer": answer["choices"][0]["text"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "LLAMA Chatbot is running. Use the /chat endpoint to interact."}
