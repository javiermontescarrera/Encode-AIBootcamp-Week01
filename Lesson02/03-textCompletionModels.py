from openai import OpenAI

client = OpenAI()

messages = [
    {
        "role": "user",
        "content": "Say this is a test",
    }
]

print(f"Prompt:\n{messages[0]['content']}\n")

models = ["gpt-3.5-turbo", "gpt-4", "gpt-4-0125-preview"]

for model in models:
    print(f"\n---\nGenerating chat completion with {model}:\n")