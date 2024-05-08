from llama_cpp import Llama
def define_model_llama_cpp_python():
    llm = Llama(
        model_path=r"C:\Users\lovej\Documents\microchatbot\app\model\mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        n_gpu_layers=-1, # Uncomment to use GPU acceleration
        seed=1337, # Uncomment to set a specific seed
        n_ctx=2048, # Uncomment to increase the context window
    )
    return llm