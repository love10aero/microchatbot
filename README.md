# Run the app
- Install the requirements
- Download the model (https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF#in-text-generation-webui)
    - pip3 install huggingface-hub>=0.17.1
    - huggingface-cli download TheBloke/Llama-2-7b-Chat-GGUF llama-2-7b-chat.Q4_K_M.gguf --local-dir . --local-dir-use-symlinks False
- run the app with uvicorn app.main:app --reload
- Make a post request with curl or postman to /chat/ endpoint and using a body like the following one:
    - {
      "question": "do your model have any characters limitation?"
    }
- It takes between 1m45s and 2m45s to answer and works without internet