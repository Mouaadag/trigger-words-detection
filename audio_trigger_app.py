import streamlit as st
import os
import numpy as np
import librosa
import tempfile
from tensorflow.keras.models import load_model

TRIGGER_WORD = "boy"  # Replace with your actual trigger word


@st.cache_resource
def load_model_once(model_path):
    return load_model(model_path)


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


def predict_audio_trigger(
    model, audio_file, sample_rate=16000, n_mfcc=13, max_pad_len=242, threshold=0.7
):
    try:
        mfccs, audio_duration = process_audio(
            audio_file, sample_rate=sample_rate, n_mfcc=n_mfcc, max_pad_len=max_pad_len
        )
        features = np.expand_dims(mfccs, axis=0)
        prediction = model.predict(features)

        contains_trigger_word = prediction[0][0] > threshold

        result = {
            "filename": os.path.basename(audio_file),
            "prediction": float(prediction[0][0]),
            "contains_trigger_word": contains_trigger_word,
        }

        if contains_trigger_word:
            start_ms, end_ms = estimate_trigger_word_time(mfccs, audio_duration)
            result["trigger_word_time"] = (
                f"The speaker said the Trigger word :  {TRIGGER_WORD}  between {start_ms:.0f} ms and {end_ms:.0f} ms"
            )

        return result

    except Exception as e:
        st.error(f"Error processing audio file: {str(e)}")
        return None


def main():
    st.title("Audio Trigger Word Detection")

    model_path = st.text_input(
        "Enter the path to your model file:", "audio_trigger_model_GRU.keras"
    )

    model = load_model_once(model_path)

    uploaded_file = st.file_uploader(
        "Choose an audio file", type=["mp3", "wav", "flac"]
    )

    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")

        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        if st.button("Detect Trigger Word"):
            result = predict_audio_trigger(model, tmp_file_path)

            if result:
                st.write(f"Filename: {result['filename']}")
                st.write(f"Prediction: {result['prediction']:.4f}")
                st.write(f"Contains trigger word: {result['contains_trigger_word']}")
                if result["contains_trigger_word"]:
                    st.write(result["trigger_word_time"])
            else:
                st.write("No prediction was made.")

        os.unlink(tmp_file_path)


if __name__ == "__main__":
    main()
