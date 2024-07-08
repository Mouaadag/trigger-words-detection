# Audio Activity Detector

This project implements an audio activity detector that triggers on the word "boy" in MP3 files.

## Project Overview

The Audio Activity Detector is designed to:
- Process MP3 audio files
- Detect the specific trigger word "boy"
- Provide a user interface using Streamlit for easy interaction

## Deployment with Streamlit

To deploy the model using Streamlit, follow these steps:

1. Clone the repository:
    ```sh
   git clone https://github.com/Mouaadag/audio-activity-detector.git
   cd audio-activity-detector

2. Install the required dependencies:
   ``` sh
   pip install -r requirements.txt

3. Run the Streamlit app:
   ``` sh
   streamlit run audio_trigger_app.py

4. Open your web browser and navigate to the URL provided by Streamlit (typically http://localhost:8501)

5. Use the interface to upload MP3 files and detect the trigger word "boy"

## Additional Information

- This project uses machine learning models to detect audio triggers
- Multiple model versions are included (GRU, L2 regularized) for comparison
- The Audio_activity_detection.ipynb notebook can be used for further analysis and model training

For any issues or suggestions, please open an issue in this repository.
