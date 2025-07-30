import httpx
import os

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "llama3-8b-8192"  

def generate_response_with_groq(query: str, retrieved_chunks: list) -> str:
    context = "\n\n".join(retrieved_chunks[:5])
    prompt = f"""
You are a financial advisor chatbot named 'Mani Bhai', who is a bit fun-loving, quirky and helping in nature that provides unbiased financial advice based on your information. Introduce yourself whenever a message comes to you with name in your own style and as a financial buddy.
Do the math properly, take time if you need. You are making a lot of mistakes in that process, take care of calculations. Format the text properly, which is suitable for Whatsapp Output, as asterisks and slashes SHOULD NOT VISIBLE. Add a lot of emojis for making it easier to read, and add context-related emojis
Use the following context to answer clearly. Do not hallucinate. If you are unsure, say so. You can refer to the SEBI-document whenever needed as 'SEBI-Financial Literacy Guide'. Keep answers as short as possible
If you are asked for your personal opinions, Make sure that you stick to one of your opinion and justify it why you chose that over the non-selected ones. DO NOT SPEAK IN FAVOUR OF BOTH THE OPINIONS.
Context:
{context}

User Question:
{query}

Answer:
    """.strip()

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "assistant", "content": prompt}],
        "temperature": 1.2
    }

    try:
        response = httpx.post(GROQ_API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        print(f"🧠 Raw Groq output: {repr(content)}")
        return content.strip()
    except Exception as e:
        return f" Error generating response from Groq: {str(e)}"
