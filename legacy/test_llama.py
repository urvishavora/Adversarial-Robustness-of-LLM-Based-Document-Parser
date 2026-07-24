import requests

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "llama3.1",
        "prompt": "Explain what a PDF parser is in one sentence.",
        "stream": False
    }
)

print(response.json()["response"])