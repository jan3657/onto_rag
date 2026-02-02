from openai import OpenAI

# 1. Setup the client to point to your forwarded port
client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="EMPTY"  # vLLM requires a key, but ignores it. We use "EMPTY".
)

# 2. Automatically get the model name (so you don't have to paste the long path)
models = client.models.list()
model_id = models.data[0].id
print(f"Connected to model: {model_id}\n")

# 3. Send a Chat Request
print("Generating response...")
completion = client.chat.completions.create(
    model=model_id,
    messages=[
        {"role": "system", "content": "You are a concise and helpful AI assistant."},
        {"role": "user", "content": "Explain the concept of 'Retrieval Augmented Generation' (RAG) to a beginner."}
    ],
    temperature=0.7,
    max_tokens=256
)

stream = client.chat.completions.create(
    model=model_id,
    messages=[{"role": "user", "content": "Explain the concept of 'Retrieval Augmented Generation' (RAG) to a beginner."}],
    stream=True  # Enable streaming
)

print("Streaming response:")
for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()