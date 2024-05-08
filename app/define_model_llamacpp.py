import os
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

def define_model_llm_chain():
    model_filename = "mistral-7b-instruct-v0.1.Q4_K_M.gguf"
    MODEL_PATH =  os.path.join("app", "model", model_filename)

    # Initialize the LlamaCpp language model with configured settings
    llm = LlamaCpp(
        model_path=MODEL_PATH,
        n_gpu_layers=40,
        n_batch=512,
        verbose=True  # Set to True for debugging
    )

    # Define the prompt template for structured queries
    template = """
    Question: {question}

    Answer:
    """
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    return llm_chain