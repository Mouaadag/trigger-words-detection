import os
import numpy as np
import librosa
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
import tempfile
from typing import List

app = FastAPI()

TRIGGER_WORD = "boy"  # Replace with your actual trigger word
MODEL_PATH = "/app/model/audio_trigger_model_GRU.keras"

# Load the model at startup
model = load_model(MODEL_PATH)


def process_audio(file_path, sample_rate=16000, n_mfcc=13, max_pad_len=242):
    audio, sr = librosa.load(file_path, sr=sample_rate)

    audio_duration = len(audio) / sr  # Duration in seconds

    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfccs = mfccs.T

    if mfccs.shape[0] > max_pad_len:
        mfccs = mfccs[:max_pad_len, :]
    else:
        pad_width = ((0, max_pad_len - mfccs.shape[0]), (0, 0))
        mfccs = np.pad(mfccs, pad_width, mode="constant")

    return mfccs, audio_duration


def estimate_trigger_word_time(mfccs, audio_duration):
    total_frames = mfccs.shape[0]
    frame_duration = audio_duration / total_frames

    start_frame = total_frames // 3
    end_frame = 2 * total_frames // 3

    start_time = start_frame * frame_duration
    end_time = end_frame * frame_duration

    return start_time * 1000, end_time * 1000  # Convert to milliseconds


@app.post("/predict/")
async def predict_trigger_word(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name

        mfccs, audio_duration = process_audio(temp_file_path)
        features = np.expand_dims(mfccs, axis=0)
        prediction = model.predict(features)

        contains_trigger_word = prediction[0][0] > 0.5

        result = {
            "filename": file.filename,
            "prediction": float(prediction[0][0]),
            "contains_trigger_word": contains_trigger_word,
        }

        if contains_trigger_word:
            start_ms, end_ms = estimate_trigger_word_time(mfccs, audio_duration)
            result["trigger_word_time"] = (
                f"The speaker said the {TRIGGER_WORD} between {start_ms:.0f} ms and {end_ms:.0f} ms"
            )

        os.unlink(temp_file_path)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {"message": "Audio Trigger Word Detection API"}


# New endpoint to update the model
@app.post("/update_model/")
async def update_model(file: UploadFile = File(...)):
    try:
        with open(MODEL_PATH, "wb") as model_file:
            model_file.write(await file.read())
        global model
        model = load_model(MODEL_PATH)
        return JSONResponse(
            content={"message": "Model updated successfully"}, status_code=200
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update model: {str(e)}")


# New endpoint to get model information
@app.get("/model_info/")
async def model_info():
    return {
        "model_name": "Audio Trigger GRU",
        "trigger_word": TRIGGER_WORD,
        "input_shape": model.input_shape,
        "output_shape": model.output_shape,
    }


# New endpoint to process multiple files
@app.post("/predict_batch/")
async def predict_trigger_word_batch(files: List[UploadFile] = File(...)):
    results = []
    for file in files:
        try:
            result = await predict_trigger_word(file)
            results.append(result)
        except HTTPException as e:
            results.append({"filename": file.filename, "error": str(e.detail)})
    return results


# New endpoint to change the trigger word
@app.post("/change_trigger_word/")
async def change_trigger_word(new_trigger_word: str):
    global TRIGGER_WORD
    TRIGGER_WORD = new_trigger_word
    return {"message": f"Trigger word changed to '{TRIGGER_WORD}'"}
