import os

import joblib
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model

IMG_SIZE = 128
MODEL_PATH = "best_covid_xray_model.h5"
LABEL_ENCODER_PATH = "label_encoder.pkl"
DEFAULT_CLASSES = ["COVID-19", "Normal", "Viral Pneumonia"]
DISPLAY_WIDTH = 480


class LabelEncoderFallback:
    def __init__(self, classes):
        self.classes_ = np.asarray(classes, dtype=str)

    def inverse_transform(self, y):
        indices = np.asarray(y, dtype=int)
        return self.classes_[indices]


@st.cache_resource
def load_label_encoder():
    if not os.path.exists(LABEL_ENCODER_PATH):
        return None, f"Label encoder file not found: {LABEL_ENCODER_PATH}"

    try:
        encoder = joblib.load(LABEL_ENCODER_PATH)
        if hasattr(encoder, "classes_"):
            return encoder, None

        if isinstance(encoder, (list, tuple, np.ndarray)):
            return LabelEncoderFallback(encoder), None

        if isinstance(encoder, dict) and "classes_" in encoder:
            return LabelEncoderFallback(encoder["classes_"]), None

        return None, "Label encoder file does not contain a valid LabelEncoder object."
    except Exception as exc:
        # Fail silently and use default class names (avoid showing this technical message to users)
        return LabelEncoderFallback(DEFAULT_CLASSES), None


@st.cache_resource
def load_resources():
    if not os.path.exists(MODEL_PATH):
        return None, None, f"Model file not found: {MODEL_PATH}", None

    try:
        tf.keras.backend.clear_session()
        model = load_model(MODEL_PATH, compile=False)
    except Exception as exc:
        return None, None, (
            "Unable to load the model file. Make sure the file is a valid Keras model "
            "saved with `model.save(...)`.\n"
            f"Error: {exc}"
        ), None

    label_encoder, label_warning = load_label_encoder()
    return model, label_encoder, None, label_warning


def preprocess_image(image: Image.Image) -> np.ndarray:
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image_array = np.asarray(image).astype("float32") / 255.0
    return np.expand_dims(image_array, axis=0)


def predict_image(image: Image.Image, model, label_encoder):
    image_array = preprocess_image(image)
    probabilities = model.predict(image_array, verbose=0)[0]
    pred_index = int(np.argmax(probabilities))
    pred_label = label_encoder.inverse_transform([pred_index])[0]
    pred_confidence = float(probabilities[pred_index])
    return pred_label, pred_confidence, probabilities


def render_header():
    st.markdown(
        """
        <style>
            .main-title {
                font-size: 2.8rem;
                font-weight: 700;
                color: #3b4cca;
                margin-bottom: 0.2rem;
            }
            .subtitle {
                font-size: 1.1rem;
                color: #555;
                margin-top: 0;
                margin-bottom: 1.5rem;
            }
            .stApp {
                background: linear-gradient(180deg, #f5f7ff 0%, #ffffff 100%);
            }
            .upload-box {
                border: 2px dashed #3b4cca;
                padding: 1rem;
                border-radius: 16px;
            }
            .upload-card {
                max-width: 920px;
                margin: 0 auto 1rem auto;
                background: linear-gradient(180deg,#ffffff,#fbfdff);
                padding: 1rem 1.25rem;
                border-radius: 14px;
                border: 1px solid rgba(11,37,69,0.06);
                box-shadow: 0 10px 30px rgba(11,37,69,0.06);
            }
            .result-card {
                background: linear-gradient(90deg, #ffffff 0%, #f7fbff 100%);
                border: 1px solid rgba(59,76,202,0.08);
                padding: 1rem;
                border-radius: 12px;
                box-shadow: 0 6px 18px rgba(59,76,202,0.06);
                margin-top: 1rem;
            }
            .result-label { font-size: 1.25rem; color: #0b2545; margin-bottom: 0.25rem; }
            .result-confidence { font-size: 1rem; color: #23527c; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="main-title">COVID-19 Chest X-ray Detection</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Upload a chest X-ray image and get a real-time prediction for Normal, COVID-19, or Viral Pneumonia.</div>',
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(
        page_title="COVID-19 X-ray Detector",
        page_icon="🩺",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    render_header()

    st.markdown("## How to use")
    st.markdown(
        """
        1. Upload a chest X-ray image in JPG, JPEG, or PNG format.  
        2. Click **Predict**.  
        3. Review the predicted class and confidence score.
        """
    )

    model, label_encoder, load_error, label_warning = load_resources()
    if load_error:
        st.error(load_error)
        return

    # Internal label encoder warnings are not shown to users for a cleaner UI

    with st.container():
        st.markdown('<div class="upload-card">', unsafe_allow_html=True)
        center_left, center_mid, center_right = st.columns([1, 2, 1])
        with center_mid:
            uploaded_file = st.file_uploader(
                "Drop or choose a chest X-ray image",
                type=["png", "jpg", "jpeg"],
                help="Best results with frontal chest X-ray (no labels/annotations).",
            )

            if uploaded_file is not None:
                try:
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Uploaded X-ray", width=DISPLAY_WIDTH, output_format="PNG")
                except Exception as exc:
                    st.error(f"Unable to open image: {exc}")
                    uploaded_file = None

            st.markdown("---")
            st.write("Tip: Crop out margins and remove overlays for better accuracy.")
        st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        if st.button("Predict", type="primary"):
            with st.spinner("Analyzing image..."):
                try:
                    label, confidence, probabilities = predict_image(image, model, label_encoder)
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")
                    return

                # Present results in a modern styled card
                st.markdown(
                    f'<div class="result-card">'
                    f'<div class="result-label">Prediction: <strong>{label}</strong></div>'
                    f'<div class="result-confidence">Confidence: {confidence * 100:.2f}%</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                class_names = getattr(label_encoder, "classes_", DEFAULT_CLASSES)
                prob_table = {
                    "Class": class_names,
                    "Probability": [f"{float(p) * 100:.2f}%" for p in probabilities],
                }

                st.markdown("### Prediction probabilities")
                st.table(prob_table)

                with st.expander("Show raw model output"):
                    st.json({
                        "predicted_label": label,
                        "confidence": confidence,
                        "probabilities": [float(p) for p in probabilities],
                    })

    

if __name__ == "__main__":
    main()