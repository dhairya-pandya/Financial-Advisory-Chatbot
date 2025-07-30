def format_whatsapp_response(answer: str) -> str:
    # Keep response readable and clean for WhatsApp
    return f" *Here's your answer:*\n\n{answer.strip()}\n\n"
def split_long_message(text, max_length=1000):
    chunks = []
    while len(text) > max_length:
        split_at = text.rfind("\n", 0, max_length)
        if split_at == -1:
            split_at = max_length
        chunks.append(text[:split_at].strip())
        text = text[split_at:].strip()
    chunks.append(text)
    return chunks
