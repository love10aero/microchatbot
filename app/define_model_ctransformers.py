import os
from ctransformers import AutoModelForCausalLM

def define_model_ctransformers():
    MODEL_PATH = os.path.join("mistral-7b-instruct-v0.1.Q4_K_M.gguf")

    # Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
    llm = AutoModelForCausalLM.from_pretrained("TheBloke/Mistral-7B-Instruct-v0.1-GGUF", model_file=MODEL_PATH, model_type="mistral", gpu_layers=50)
    return llm