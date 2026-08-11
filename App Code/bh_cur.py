from pathlib import Path
import textwrap
import numpy as np
from PIL import Image, ImageOps
import pandas as pd
import streamlit as st
import tensorflow as tf


# --------------------------------------------------
# 1. Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Bahraini Currency Recognition",
    page_icon="💵",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# 2. Styling
# --------------------------------------------------

st.markdown(
    """
    <style>

    .main {
        background-color: #f8f9fa;
    }

    .metric-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border: 1px solid #e2e8f0;
        text-align: center;
        margin-bottom: 15px;
    }

    .stButton > button {
        width: 100%;
        background-color: #2563eb;
        color: white;
        border-radius: 8px;
        height: 3em;
        font-weight: 600;
        border: none;
    }

    .stButton > button:hover {
        background-color: #1d4ed8;
        color: white;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# 3. Session State
# --------------------------------------------------

if "detected_currency" not in st.session_state:
    st.session_state.detected_currency = []


# --------------------------------------------------
# 4. Load Model
# --------------------------------------------------

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "currency_model4.keras"
)


def load_my_model():

    if not MODEL_PATH.exists():

        st.error(
            f"Model file not found at:\n{MODEL_PATH}"
        )

        return None

    try:

        return tf.keras.models.load_model(
            str(MODEL_PATH)
        )

    except Exception as e:

        st.error(
            f"Error loading {MODEL_PATH.name}: {e}"
        )

        return None


model = load_my_model()


# --------------------------------------------------
# 5. Class Mapping
# --------------------------------------------------

CURRENCY_CLASSES = {

    0: {
        "name": "0.5 BD",
        "value": 0.500
    },

    1: {
        "name": "1 BD",
        "value": 1.000
    },

    2: {
        "name": "10 BD",
        "value": 10.000
    },

    3: {
        "name": "100 Fils",
        "value": 0.100
    },

    4: {
        "name": "5 BD",
        "value": 5.000
    },

    5: {
        "name": "50 Fils",
        "value": 0.050
    }

}


# --------------------------------------------------
# 6. Sidebar
# --------------------------------------------------

with st.sidebar:

    st.title("💰 Currency Calculator")

    st.info(
        "Upload or capture an image of Bahraini currency. "
        "The model will identify its denomination and allow "
        "you to add it to the running total."
    )

    st.divider()

    st.subheader("Accumulated Total")

    total_val = sum(
        item["Value"]
        for item in st.session_state.detected_currency
    )

    st.metric(
        label="Total Amount",
        value=f"{total_val:.3f} BHD"
    )

    if st.button("Reset Calculator"):

        st.session_state.detected_currency = []

        if "last_detected" in st.session_state:
            del st.session_state["last_detected"]

        st.rerun()


# --------------------------------------------------
# 7. Main Page
# --------------------------------------------------

st.title("Bahraini Currency Identifier")

st.caption("Powered by TensorFlow and Streamlit")

st.write("---")

col1, col2 = st.columns(
    [1, 1],
    gap="large"
)


# --------------------------------------------------
# 8. Image Input
# --------------------------------------------------

with col1:

    st.subheader("📷 Input")

    input_type = st.radio(
        "Select Input Method:",
        [
            "Upload Image",
            "Take Photo via Camera"
        ],
        horizontal=True
    )

    image = None


    if input_type == "Upload Image":

        uploaded_file = st.file_uploader(
            "Choose a currency image...",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:

            image = Image.open(
                uploaded_file
            )


    else:

        camera_file = st.camera_input(
            "Take a photo of the currency"
        )

        if camera_file is not None:

            image = Image.open(
                camera_file
            )


    if image is not None:

        st.image(
            image,
            caption="Selected Image",
            use_container_width=True
        )


# --------------------------------------------------
# 9. Prediction
# --------------------------------------------------

with col2:

    st.subheader("📊 Prediction")

    if image is None:

        st.warning(
            "Please upload or capture an image."
        )

    else:

        # Fix rotation from camera metadata
        image = ImageOps.exif_transpose(
            image
        )

        # Convert to RGB
        image = image.convert(
            "RGB"
        )

        # Resize to same size used during training
        image_resized = image.resize(
            (224, 224),
            Image.Resampling.BILINEAR
        )

        # Convert to NumPy
        image_array = np.array(
            image_resized,
            dtype=np.float32
        )

        # IMPORTANT:
        # Do NOT divide by 255 here.
        # The model already contains Rescaling(1./255).

        img_processed = image_array

        # Add batch dimension
        img_processed = np.expand_dims(
            img_processed,
            axis=0
        )


        if st.button(
            "🚀 Identify Currency"
        ):

            if model is not None:

                with st.spinner(
                    "Analyzing image..."
                ):

                    predictions = model.predict(
                        img_processed,
                        verbose=0
                    )

                    predicted_idx = int(
                        np.argmax(
                            predictions[0]
                        )
                    )

                    confidence = float(
                        np.max(
                            predictions[0]
                        ) * 100
                    )

                    currency_info = (
                        CURRENCY_CLASSES[
                            predicted_idx
                        ]
                    )

                    st.session_state[
                        "last_detected"
                    ] = {

                        "Currency":
                            currency_info["name"],

                        "Value":
                            currency_info["value"],

                        "Confidence":
                            confidence
                    }

            else:

                st.error(
                    "Model could not be loaded."
                )

# --------------------------------------------------
# 10. Show Prediction
# --------------------------------------------------

if "last_detected" in st.session_state:

    last = st.session_state["last_detected"]

    html_card = textwrap.dedent(f"""
        <div class="metric-card">
            <h4 style="color:#64748b; margin:0;">
                Detected Currency
            </h4>
            <h1 style="color:#2563eb; font-size:2.2rem; margin:10px 0;">
                {last["Currency"]}
            </h1>
            <p style="color:#475569; margin:0;">
                Value: <strong>{last["Value"]:.3f} BHD</strong>
            </p>
        </div>
    """)

    st.markdown(html_card, unsafe_allow_html=True)

    st.write("**Confidence Score:**")
    st.progress(last["Confidence"] / 100)
    st.caption(f'Model Confidence: **{last["Confidence"]:.2f}%**')

    st.write("---")

    if st.button("➕ Add to Total"):
        st.session_state.detected_currency.append({
            "Currency": last["Currency"],
            "Value": last["Value"]
        })
        st.success(f'Added {last["Currency"]} ({last["Value"]:.3f} BHD)')
        st.rerun()