from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

app = FastAPI()

# Load the LlamaCpp language model
llm = LlamaCpp(
    model_path=r"C:\Users\lovej\Documents\microchatbot\app\model\llama-2-7b-chat.Q4_K_M.gguf",
    n_gpu_layers=40,
    n_batch=512,
    verbose=False
)

# Define the prompt template
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
        answer = llm_chain.run(query.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "LLAMA Chatbot is running. Use the /chat endpoint to interact."}
