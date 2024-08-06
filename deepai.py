import streamlit as st
import numpy as np
import librosa
import tensorflow as tf

# Define a simple CNN model for demonstration purposes
def create_model(input_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(6, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Dummy function to simulate loading a pre-trained model
def load_pretrained_model(input_shape):
    model = create_model(input_shape)
    return model

# Function to extract MFCC features from audio
def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=44100)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs = np.expand_dims(mfccs, axis=-1)
    mfccs = np.expand_dims(mfccs, axis=0)
    return mfccs

# Function to predict emotion from audio features
def predict_emotion(model, mfccs):
    predictions = model.predict(mfccs)
    return np.argmax(predictions)

# Streamlit interface
st.title("Audio Emotion Recognition")
st.write("Upload an audio file to recognize the emotion")

uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    with open("temp_audio.mp3", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Extract features
    mfccs = extract_features("temp_audio.mp3")
    
    # Load the pre-trained model with the correct input shape
    model = load_pretrained_model(mfccs.shape[1:])
    
    # Predict emotion
    emotion = predict_emotion(model, mfccs)
    emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise"]
    
    st.write(f"Predicted Emotion: {emotions[emotion]}")
    
    # Display audio player
    st.audio(uploaded_file)



