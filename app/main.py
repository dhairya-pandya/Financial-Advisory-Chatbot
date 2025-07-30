from fastapi import FastAPI, Form, UploadFile, BackgroundTasks
from fastapi.responses import PlainTextResponse
from twilio.twiml.messaging_response import MessagingResponse
import httpx, tempfile, subprocess, os, speech_recognition as sr

from app.rag import retrieve_chunks
from app.llm_gen import generate_response_with_groq
from app.utils import format_whatsapp_response, split_long_message
import asyncio

app = FastAPI()
http_client = httpx.AsyncClient()

def download_and_transcribe(media_url: str, twilio_auth: tuple) -> str:
    resp = httpx.get(media_url, auth=twilio_auth, follow_redirects=True)
    resp.raise_for_status()

    with tempfile.TemporaryDirectory() as td:
        ogg_path = os.path.join(td, "voice.ogg")
        wav_path = os.path.join(td, "voice.wav")
        with open(ogg_path, "wb") as f:
            f.write(resp.content)

        
        subprocess.run(
            ["ffmpeg", "-y", "-i", ogg_path, "-ar", "16000", "-ac", "1", wav_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        r = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio = r.record(source)
        try:
            return r.recognize_google(audio)  
        except sr.UnknownValueError:
            return ""


@app.post("/webhook")
async def whatsapp_webhook(
    Body: str = Form(""),
    NumMedia: str = Form("0"),
    MediaUrl0: str = Form(None),
    MediaContentType0: str = Form(None),
):
    try:
        query_text = Body.strip()

        # voice note handling
        if (not query_text) and NumMedia != "0" and MediaContentType0.startswith("audio"):
            print("🎙 Voice note detected → downloading…")
            twilio_auth = ("TWILIO_AUTH", 'TWILIO_SSID')
            query_text = await asyncio.to_thread(
                download_and_transcribe, MediaUrl0, twilio_auth
            )
            if not query_text:
                raise ValueError("Could not transcribe audio.")

        print(f"🔔 Query: {query_text}")

        chunks = retrieve_chunks(query_text)
        answer = generate_response_with_groq(query_text, chunks)
        formatted = format_whatsapp_response(answer)
        parts = split_long_message(formatted)
        print(f"🔔 Answer: {formatted}")
        resp = MessagingResponse()
        for p in parts:
            resp.message(p)

        return PlainTextResponse(str(resp), media_type="application/xml")

    except Exception as e:
        print(f"💥 Error: {e}")
        resp = MessagingResponse()
        resp.message("❌ An error occurred. Please try again later.")
        return PlainTextResponse(str(resp), media_type="application/xml")

