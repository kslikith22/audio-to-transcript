import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
import numpy as np
import librosa
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from transformers import Wav2Vec2Tokenizer, TFWav2Vec2ForCTC , AutoProcessor
import io

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = TFWav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

#! will use in future since large amount of downloading model is required (around 3.9 gb) 
# processor = AutoProcessor.from_pretrained("facebook/mms-1b-all")
# model = TFWav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all")

async def load_audio_from_upload(file: UploadFile):
    try:
        content = await file.read()
        return librosa.load(io.BytesIO(content), sr=16000)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Error loading audio file")

def tokenize_audio(audio_data):
    try:
        inputs = tokenizer(audio_data, return_tensors="tf", padding="longest")
        return tf.convert_to_tensor(inputs.input_values)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error tokenizing audio")

def transcribe_audio(input_values):
    try:
        logits = model(input_values).logits
        predicted_ids = tf.argmax(logits, axis=-1)
        transcription = tokenizer.batch_decode(predicted_ids)[0]
        return transcription
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error transcribing audio")

@app.post("/convert")
async def convert_audio(file: UploadFile = File(...)):
    try:
        audio_data, _ = await load_audio_from_upload(file)
        input_values = tokenize_audio(audio_data)
        transcription = transcribe_audio(input_values)
        return {"transcription": transcription}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
