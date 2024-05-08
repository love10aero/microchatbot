from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

app = FastAPI()

MODEL_PATH = ".\model\mistral-7b-instruct-v0.1.Q4_0.gguf"

# Initialize the LlamaCpp language model with configured settings
llm = LlamaCpp(
    model_path=MODEL_PATH,
    n_gpu_layers=40,
    n_batch=512,
    verbose=False  # Set to True for debugging
)

# Define the prompt template for structured queries
template = """
Question: {question}

Answer:
"""
prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = LLMChain(prompt=prompt, llm=llm)

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
