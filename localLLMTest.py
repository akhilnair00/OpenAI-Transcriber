###########################################################

# Test 1

# from transformers import pipeline

# text_input = "Who are you?"

# pipe = pipeline("text-generation", model="gpt2")

# output = pipe(text_input)
# print(output)

# Results: Run's poorly.

###########################################################

# Test 2

import sys
from langchain_community.llms import LlamaCpp

verbose = False

llm = LlamaCpp(
    model_path="C:/Users/akhil/OneDrive/Documents/Creative Endeavour/OpenAI Transcriber/synthia-7b-v2.0-16k.Q4_K_M.gguf",
    n_ctx=4096,
    n_gpu_layers=32,
    n_batch=1024,
    f16_kv=True,
    verbose=verbose,
)

question = "I am feeling down, what can I do?"

output = llm(
    question,
    max_tokens=4096,
    temperature=0.2,
    top_p=0.1
)

print({output})

###########################################################