"""
app.py — Streamlit deployment for the EuroSAT dual-mode classifier.

Two independent inference paths:
1. RGB (3-band) satellite images
2. 13-band multispectral Sentinel-2 GeoTIFFs

Both use the frozen normalization_stats.json produced during training.
"""

import json
from pathlib import Path
import tempfile

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
import tifffile
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import transforms

from model import SEResEuroNet


# ==========================================================================
# PAGE CONFIG
# ==========================================================================

st.set_page_config(
    page_title="EuroSAT Land Cover Classifier",
    page_icon="🛰️",
    layout="wide",
)


# ==========================================================================
# CONSTANTS & FROZEN ARTIFACTS
# ==========================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = 64

CLASS_NAMES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]

BASE_DIR = Path(__file__).parent

STATS_PATH = BASE_DIR / "normalization_stats.json"

with open(STATS_PATH, "r") as f:
    STATS = json.load(f)

RGB_MEAN = STATS["rgb"]["mean"]
RGB_STD = STATS["rgb"]["std"]

MS_MEAN = torch.tensor(
    STATS["multispectral"]["mean"],
    dtype=torch.float32
).view(-1, 1, 1)

MS_STD = torch.tensor(
    STATS["multispectral"]["std"],
    dtype=torch.float32
).view(-1, 1, 1)


# ==========================================================================
# MODEL LOADING
# ==========================================================================

@st.cache_resource
def load_model(checkpoint_path, in_channels):
    model = SEResEuroNet(
        in_channels=in_channels,
        num_classes=len(CLASS_NAMES)
    )

    state_dict = torch.load(
        checkpoint_path,
        map_location=DEVICE
    )

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model

# ==========================================================================
# HUGGING FACE MODEL CHECKPOINTS
# ==========================================================================

HF_REPO = "MallikarjunJadi/eurosat-land-cover-models"

RGB_CHECKPOINT = hf_hub_download(
    repo_id=HF_REPO,
    filename="rgb_model_best.pt"
)

MS_CHECKPOINT = hf_hub_download(
    repo_id=HF_REPO,
    filename="multispectral_model_best.pt"
)


rgb_model = load_model(
    RGB_CHECKPOINT,
    in_channels=3
)

ms_model = load_model(
    MS_CHECKPOINT,
    in_channels=13
)


# ==========================================================================
# RGB PREPROCESSING
# ==========================================================================

rgb_inference_transform = transforms.Compose([
    transforms.Resize(
        (IMG_SIZE, IMG_SIZE),
        interpolation=transforms.InterpolationMode.BILINEAR
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=RGB_MEAN,
        std=RGB_STD
    ),
])


def predict_rgb(image):
    image = image.convert("RGB")

    tensor = (
        rgb_inference_transform(image)
        .unsqueeze(0)
        .to(DEVICE)
    )

    with torch.no_grad():
        logits = rgb_model(tensor)

        probs = (
            F.softmax(logits, dim=1)
            .squeeze(0)
            .cpu()
            .numpy()
        )

    return {
        CLASS_NAMES[i]: float(probs[i])
        for i in range(len(CLASS_NAMES))
    }


# ==========================================================================
# MULTISPECTRAL FUNCTIONS
# ==========================================================================

def _read_tif_bands(file_path):
    """
    Reads a 13-band GeoTIFF.

    Supports:
        (13, H, W)
        (H, W, 13)

    Returns:
        (13, H, W)
    """

    arr = tifffile.imread(file_path).astype(np.float32)

    if (
        arr.ndim == 3
        and arr.shape[-1] == 13
        and arr.shape[0] != 13
    ):
        arr = arr.transpose(2, 0, 1)

    return arr


def _resize_bands(arr, size):
    """
    Resize all 13 bands using bilinear interpolation.
    """

    tensor = torch.from_numpy(arr).unsqueeze(0)

    tensor = F.interpolate(
        tensor,
        size=(size, size),
        mode="bilinear",
        align_corners=False
    )

    return tensor.squeeze(0)


def _false_color_preview(arr):
    """
    Creates a natural-color preview using:
        B4 = Red
        B3 = Green
        B2 = Blue

    Internally:
        0-indexed bands [3, 2, 1]
    """

    rgb = arr[[3, 2, 1], :, :]

    p99 = np.percentile(rgb, 99)

    if p99 <= 0:
        p99 = 1.0

    rgb = np.clip(rgb / p99, 0, 1)

    rgb = (
        rgb.transpose(1, 2, 0) * 255
    ).astype(np.uint8)

    return rgb


def predict_multispectral(file_path):

    raw = _read_tif_bands(file_path)

    # Validate number of bands
    if raw.ndim != 3:
        raise ValueError(
            f"Invalid GeoTIFF shape: {raw.shape}"
        )

    if raw.shape[0] != 13:
        raise ValueError(
            f"Expected a 13-band Sentinel-2 GeoTIFF, "
            f"but detected {raw.shape[0]} bands."
        )

    # Preview
    preview = _false_color_preview(raw)

    # Resize
    tensor = _resize_bands(
        raw,
        IMG_SIZE
    )

    # Frozen training normalization
    tensor = (
        tensor - MS_MEAN
    ) / MS_STD

    tensor = (
        tensor
        .unsqueeze(0)
        .to(DEVICE)
    )

    # Inference
    with torch.no_grad():

        logits = ms_model(tensor)

        probs = (
            F.softmax(logits, dim=1)
            .squeeze(0)
            .cpu()
            .numpy()
        )

    prediction = {
        CLASS_NAMES[i]: float(probs[i])
        for i in range(len(CLASS_NAMES))
    }

    return prediction, preview


# ==========================================================================
# HEADER
# ==========================================================================

st.title("🛰️ EuroSAT Land Cover Classifier")

st.markdown(
    """
    Classify satellite imagery into **10 land-cover categories**
    using deep learning.

    Choose between **RGB** imagery or a **13-band Sentinel-2
    multispectral GeoTIFF**.
    """
)


# ==========================================================================
# SIDEBAR
# ==========================================================================

with st.sidebar:

    st.header("⚙️ Model Information")

    st.write(
        f"**Device:** `{DEVICE}`"
    )

    st.write(
        "**Input Size:** 64 × 64"
    )

    st.write(
        "**Classes:** 10"
    )

    st.write(
        "**RGB Model:** 3 channels"
    )

    st.write(
        "**Multispectral Model:** 13 channels"
    )

    st.divider()

    st.caption(
        "Normalization statistics are frozen from training."
    )


# ==========================================================================
# TABS
# ==========================================================================

rgb_tab, ms_tab = st.tabs(
    [
        "🖼️ RGB Classification",
        "🌈 13-Band Multispectral",
    ]
)


# ==========================================================================
# RGB TAB
# ==========================================================================

with rgb_tab:

    st.subheader("RGB Satellite Image")

    st.info(
        "Upload a normal RGB satellite image "
        "(.png, .jpg, .jpeg)."
    )

    uploaded_rgb = st.file_uploader(
        "Choose an RGB image",
        type=["png", "jpg", "jpeg"],
        key="rgb_uploader",
    )

    if uploaded_rgb is not None:

        image = Image.open(uploaded_rgb)

        col1, col2 = st.columns(2)

        with col1:

            st.image(
                image,
                caption="Uploaded RGB Image",
                use_container_width=True
            )

        with col2:

            with st.spinner("Running RGB model..."):

                probabilities = predict_rgb(image)

            sorted_probs = sorted(
                probabilities.items(),
                key=lambda x: x[1],
                reverse=True
            )

            predicted_class = sorted_probs[0][0]
            confidence = sorted_probs[0][1]

            st.success(
                f"Prediction: **{predicted_class}**"
            )

            st.metric(
                "Confidence",
                f"{confidence * 100:.2f}%"
            )

            st.write("### Top Predictions")

            for class_name, probability in sorted_probs[:5]:

                st.write(
                    f"**{class_name}** — "
                    f"{probability * 100:.2f}%"
                )

                st.progress(
                    float(probability)
                )


# ==========================================================================
# MULTISPECTRAL TAB
# ==========================================================================

with ms_tab:

    st.subheader(
        "13-Band Sentinel-2 Multispectral Image"
    )

    st.info(
        "Upload a genuine 13-band Sentinel-2 GeoTIFF "
        "(.tif / .tiff)."
    )

    uploaded_tif = st.file_uploader(
        "Choose a 13-band GeoTIFF",
        type=["tif", "tiff"],
        key="ms_uploader",
    )

    if uploaded_tif is not None:

        try:

            # Streamlit UploadedFile is not a normal filesystem path.
            # Save temporarily so tifffile can read it.
            with tempfile.NamedTemporaryFile(
                suffix=".tif",
                delete=False
            ) as tmp:

                tmp.write(
                    uploaded_tif.getbuffer()
                )

                temp_path = tmp.name

            # Read raw image first for validation
            raw = _read_tif_bands(temp_path)

            if raw.ndim != 3:

                st.error(
                    f"Invalid GeoTIFF dimensions: {raw.shape}"
                )

            elif raw.shape[0] != 13:

                st.error(
                    f"""
                    ❌ Invalid input.

                    Expected **13 bands**, but this file
                    contains **{raw.shape[0]} bands**.

                    Please upload a 13-band Sentinel-2 GeoTIFF.
                    """
                )

            else:

                st.success(
                    "✅ Valid 13-band Sentinel-2 GeoTIFF detected."
                )

                st.write(
                    f"**Image dimensions:** "
                    f"{raw.shape[1]} × {raw.shape[2]}"
                )

                st.write(
                    "**Bands detected:** 13"
                )

                preview = _false_color_preview(raw)

                col1, col2 = st.columns(2)

                with col1:

                    st.image(
                        preview,
                        caption="Natural-color preview (B4-B3-B2)",
                        use_container_width=True
                    )

                with col2:

                    with st.spinner(
                        "Running 13-band model..."
                    ):

                        probabilities, _ = (
                            predict_multispectral(
                                temp_path
                            )
                        )

                    sorted_probs = sorted(
                        probabilities.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )

                    predicted_class = sorted_probs[0][0]
                    confidence = sorted_probs[0][1]

                    st.success(
                        f"Prediction: **{predicted_class}**"
                    )

                    st.metric(
                        "Confidence",
                        f"{confidence * 100:.2f}%"
                    )

                    st.write(
                        "### Top Predictions"
                    )

                    for class_name, probability in sorted_probs[:5]:

                        st.write(
                            f"**{class_name}** — "
                            f"{probability * 100:.2f}%"
                        )

                        st.progress(
                            float(probability)
                        )

        except Exception as e:

            st.error(
                f"Unable to process this GeoTIFF: {e}"
            )


# ==========================================================================
# FOOTER
# ==========================================================================

st.divider()

st.caption(
    "EuroSAT Dual-Mode Land Cover Classifier • "
    "PyTorch + SE-ResNet • "
    "Frozen training normalization"
)