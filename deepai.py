import sys
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import cv2
import librosa
import librosa.display
from tensorflow.keras.models import load_model
import warnings
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import joblib

# Load models
# model = load_model(".venv/audio.h5", compile=False)
# tmodel = load_model("tmodel_all.h5")

# Constants
CAT6 = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise']
CAT3 = ["positive", "negative"]
emotion_combinations = {'depression': ['sad', 'fear'], 'anxiety': ['fear', 'surprise'], 'dementia': ['confusion', 'disgust']}

# Page settings
st.set_page_config(layout="wide")

max_width = 1000
padding_top = 0
padding_right = "20%"
padding_left = "10%"
padding_bottom = 0
COLOR = "#1f1f2e"
BACKGROUND_COLOR = "#d1d1e0"
st.markdown(
    f"""
<style>
    .reportview-container .main .block-container{{
        max-width: {max_width}px;
        padding-top: {padding_top}rem;
        padding-right: {padding_right}rem;
        padding-left: {padding_left}rem;
        padding-bottom: {padding_bottom}rem;
    }}
    .reportview-container .main {{
        color: {COLOR};
        background-color: {BACKGROUND_COLOR};
    }}
</style>
""",
    unsafe_allow_html=True,
)

@st.cache
def save_audio(file):
    with open(os.path.join("audio", file.name), "wb") as f:
        f.write(file.getbuffer())

@st.cache
def get_melspec(audio):
    y, sr = librosa.load(audio, sr=44100)
    X = librosa.stft(y)
    Xdb = librosa.amplitude_to_db(abs(X))
    img = np.stack((Xdb,) * 3, -1)
    img = img.astype(np.uint8)
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayImage = cv2.resize(grayImage, (224, 224))
    rgbImage = np.repeat(grayImage[..., np.newaxis], 3, -1)
    return rgbImage, Xdb

@st.cache
def get_mfccs(audio, limit):
    y, sr = librosa.load(audio, sr=44100)
    a = librosa.feature.mfcc(y=y, sr=44100, n_mfcc=162)
    if a.shape[1] > limit:
        mfccs = a[:, :limit]
    elif a.shape[1] < limit:
        mfccs = np.zeros((a.shape[0], limit))
        mfccs[:, :a.shape[1]] = a
    return mfccs

@st.cache
def get_title(predictions, categories=CAT6):
    detected_emotion = categories[predictions.argmax()]
    mental_states = get_mental_state(predictions, categories)
    mental_states_str = ",".join(mental_states) if mental_states else "none"
    title = f"Detected emotion: {detected_emotion} - {predictions.max() * 100:.2f}%\nPotential mental states: {mental_states_str}"
    return title

@st.cache
def plot_emotions(fig, data6, data3=None, title="Detected emotion", categories6=CAT6, categories3=CAT3):
    color_dict = {
        "fear": "grey",
        "positive": "green",
        "angry": "green",
        "happy": "orange",
        "sad": "purple",
        "negative": "red",
        "disgust": "red",
        "surprise": "lightblue"
    }

    if data3 is None:
        pos = data6[3] + data6[5]
        neg = data6[0] + data6[1] + data6[2] + data6[4]
        data3 = np.array([pos, neg])

    ind = categories6[data6.argmax()]
    color6 = color_dict[ind]

    data6 = list(data6)
    n = len(data6)
    data6 += data6[:1]
    angles6 = [i / float(n) * 2 * np.pi for i in range(n)]
    angles6 += angles6[:1]

    ind = categories3[data3.argmax()]
    color3 = color_dict[ind]

    data3 = list(data3)
    n = len(data3)
    data3 += data3[:1]
    angles3 = [i / float(n) * 2 * np.pi for i in range(n)]
    angles3 += angles3[:1]

    fig.set_facecolor('#d1d1e0')
    ax = plt.subplot(122, polar="True")
    plt.polar(angles6, data6, color=color6)
    plt.fill(angles6, data6, facecolor=color6, alpha=0.25)

    ax.spines['polar'].set_color('lightgrey')
    ax.set_theta_offset(np.pi / 3)
    ax.set_theta_direction(-1)
    plt.xticks(angles6[:-1], categories6)
    ax.set_rlabel_position(0)
    plt.yticks([0, .25, .5, .75, 1], color="grey", size=8)
    plt.ylim(0, 1)

    ax = plt.subplot(121, polar="True")
    plt.polar(angles3, data3, color=color3, linewidth=2, linestyle="--", alpha=.8)
    plt.fill(angles3, data3, facecolor=color3, alpha=0.25)

    ax.spines['polar'].set_color('lightgrey')
    ax.set_theta_offset(np.pi / 6)
    ax.set_theta_direction(-1)
    plt.xticks(angles3[:-1], categories3)
    ax.set_rlabel_position(0)
    plt.yticks([0, .25, .5, .75, 1], color="grey", size=8)
    plt.ylim(0, 1)
    plt.suptitle(title)
    plt.subplots_adjust(top=0.75)

if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

def noise(data):
    noise_amp = 0.035 * np.random.uniform() * np.amax(data)
    data = data + noise_amp * np.random.normal(size=data.shape[0])
    return data

def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate=0.8)

def shift(data):
    shift_range = int(np.random.uniform(low=-5, high=5) * 1000)
    return np.roll(data, shift_range)

def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sr=sampling_rate, n_steps=0.7)

def extract_features(data, sample_rate):
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))

    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft))

    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc))

    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))

    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))
    return result

def get_features(data):
    data, sample_rate = librosa.load(data, duration=2.5, offset=0.6)

    res1 = extract_features(data, sample_rate)
    result = np.array(res1)

    noise_data = noise(data)
    res2 = extract_features(noise_data, sample_rate)
    result = np.vstack((result, res2))

    new_data = stretch(data)
    data_stretch_pitch = pitch(new_data, sample_rate)
    res3 = extract_features(data_stretch_pitch, sample_rate)
    result = np.vstack((result, res3))

    return result

def get_mental_state(pred, categories=CAT6):
    detected_emotion = categories[pred.argmax()]
    results = []
    for state, emotions in emotion_combinations.items():
        if detected_emotion in emotions:
            results.append(state)
    return results

def main():
    st.title("Neu-Free emotion analysis for e-care buddy hearing product")
    st.sidebar.markdown("## Use the menu to navigate on the site")

    menu = ["Upload audio", "Dataset analysis", "Deployment", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Upload audio":
        st.subheader("Emotion detection")

        audio_file = st.file_uploader("Choose an audio...", type=["wav", "mp3", "ogg"])

        if audio_file:
            save_audio(audio_file)
            melspec_img, xdb = get_melspec(audio_file)
            mfccs = get_mfccs(audio_file, limit=100)
            ftrs = get_features(audio_file)
            st.image(melspec_img, width=400)
            st.audio(audio_file)

            ftrs = np.expand_dims(ftrs, axis=-1)
            pred6 = model.predict(ftrs).flatten()
            pred3 = tmodel.predict(mfccs).flatten()

            title = get_title(pred6)
            fig = plt.figure(figsize=(10, 6))
            plot_emotions(fig, pred6, pred3, title=title)
            st.pyplot(fig)

            st.write("### Model predictions")
            st.write(f"Emotion categories: {CAT6}")
            st.write(f"Emotion probabilities: {pred6}")
            st.write(f"3 Category emotions: {pred3}")

    elif choice == "Dataset analysis":
        st.subheader("Data analysis for the uploaded audio")

    elif choice == "Deployment":
        st.subheader("Deploy your model using streamlit")

    else:
        st.subheader("About")
        st.write("This app uses ML models to analyze emotions from audio files.")

if __name__ == "__main__":
    main()
