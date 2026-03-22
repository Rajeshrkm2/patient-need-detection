import os
import tempfile
import streamlit as st
from emotion_package import NeedDetector

st.set_page_config(page_title="Patient Need Detection", layout="centered")

st.title("Patient Need Detection System")
st.write("Upload a facial image to predict the patient's need.")

KNOWN_FACES_PATH = os.path.join("emotion_package", "known_faces")

try:
    detector = NeedDetector(KNOWN_FACES_PATH)
except Exception as e:
    st.error(f"Error loading detector: {e}")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload patient image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_path = tmp_file.name

    predicted_need, matched_file, score = detector.predict_need(temp_path)

    st.subheader("Prediction Result")
    st.write(f"**Matched Reference Image:** {matched_file}")
    st.write(f"**Predicted Need:** {predicted_need}")
    st.write(f"**Similarity Score:** {score}")

    if os.path.exists(temp_path):
        os.remove(temp_path)