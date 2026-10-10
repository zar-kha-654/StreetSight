from report_generator import generate_evidence_pdf
import hashlib
from datetime import datetime, timezone

import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from huggingface_hub import hf_hub_download
from ultralytics import YOLO


# -----------------------------
# PAGE CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="StreetSight | Road Damage Evidence",
    page_icon="🛣️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #F5F7FA;
    }
    h1, h2, h3 {
        color: #10233F;
    }
    .hero {
        background: #10233F;
        padding: 28px;
        border-radius: 16px;
        color: white;
        margin-bottom: 20px;
    }
    .hero h1 {
        color: white;
        margin-bottom: 8px;
    }
    .hero p {
        color: #E5EAF2;
        font-size: 17px;
    }
    .note {
        color: #5B6472;
        font-size: 13px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# MODEL LOADING
# -----------------------------
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="DanielsStulpe/pothole-detection",
        filename="yolo26_best.pt",
    )
    return YOLO(model_path)


# -----------------------------
# HEADER
# -----------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🛣️ StreetSight</h1>
        <p>
        See the damage. Verify the evidence.
        Report the problem.
        </p>
        From road photograph to human-reviewed evidence.
    </div>
    """,
    unsafe_allow_html=True,
)

st.write(
    "Upload a road photograph to identify potential potholes. "
    "Review every prediction before preparing evidence."
)

st.info(
    "Current model scope: pothole detection only. "
    "A prediction is not an official inspection or confirmation."
)


# -----------------------------
# INPUTS
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload a road photograph",
    type=["jpg", "jpeg", "png"],
    help="Use a clear image showing the road surface.",
)


st.subheader("Where was the damage observed?")

road_name = st.text_input(
    "Road or street name",
    placeholder="e.g. Kashmir Highway",
)

area_name = st.text_input(
    "Area or neighborhood",
    placeholder="e.g. G-10",
)

city_name = st.text_input(
    "City",
    value="Islamabad",
)

landmark = st.text_input(
    "Nearby landmark (optional)",
    placeholder="e.g. Near a bus stop or market",
)

location_parts = [
    value.strip()
    for value in [
        road_name,
        area_name,
        city_name,
        landmark,
    ]
    if value and value.strip()
]

location = ", ".join(location_parts)


confidence = st.slider(
    "Detection confidence threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.35,
    step=0.05,
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        image = ImageOps.exif_transpose(image).convert("RGB")

        # Basic upload safety and resource limits.
        if image.width * image.height > 20_000_000:
            st.error(
                "This image is too large. Please upload "
                "an image with at most 20 million pixels."
            )
            st.stop()

        image_hash = hashlib.sha256(
            image.tobytes()
        ).hexdigest()[:12]

        st.subheader("Uploaded photograph")
        st.image(image, use_container_width=True)

        analyze = st.button(
            "Analyze image",
            type="primary",
            use_container_width=True,
        )

        if analyze:
            try:
                with st.spinner(
                    "Loading the detector and analyzing your image..."
                ):
                    model = load_model()

                    results = model.predict(
                        source=np.array(image),
                        conf=confidence,
                        imgsz=640,
                        device="cpu",
                        verbose=False,
                    )

                    result = results[0]

                    # Ultralytics plot() returns a BGR image.
                    annotated_bgr = result.plot()
                    annotated_rgb = annotated_bgr[:, :, ::-1]

                    detections = []

                    if result.boxes is not None:
                        for index, box in enumerate(result.boxes):
                            class_id = int(
                                box.cls[0].item()
                            )
                            class_name = result.names[class_id]
                            score = float(
                                box.conf[0].item()
                            )
                            coordinates = [
                                round(float(value), 1)
                                for value in box.xyxy[0].tolist()
                            ]

                            detections.append(
                                {
                                    "id": index,
                                    "class": class_name,
                                    "confidence": score,
                                    "box": coordinates,
                                }
                            )

                    st.session_state["analysis"] = {
                        "image_hash": image_hash,
                        "original": image.copy(),
                        "annotated": Image.fromarray(
                            annotated_rgb
                        ),
                        "detections": detections,
                        
                        "location": location,
                        "road_name": road_name.strip(),
                        "area_name": area_name.strip(),
                        "city_name": city_name.strip(),
                        "landmark": landmark.strip(),

                        "timestamp": datetime.now(
                            timezone.utc
                        ).isoformat(timespec="seconds"),
                    }

            except Exception as error:
                st.error(
                    "Image analysis failed. Please try again. "
                    "If the problem continues, check the app logs."
                )
                st.caption(
                    f"Technical detail: {type(error).__name__}: "
                    f"{str(error)[:300]}"
                )

    except Exception:
        st.error(
            "We couldn't read this image. Please upload a "
            "valid JPG, JPEG, or PNG file."
        )



# -----------------------------
# RESULTS AND HUMAN REVIEW
# -----------------------------
analysis = st.session_state.get("analysis")

if analysis is not None:
    # Display results only for the currently uploaded image.
    if uploaded_file is not None:
        try:
            current_image = ImageOps.exif_transpose(
                Image.open(uploaded_file)
            ).convert("RGB")

            current_hash = hashlib.sha256(
                current_image.tobytes()
            ).hexdigest()[:12]

        except Exception:
            current_hash = None

        if current_hash == analysis["image_hash"]:
            st.divider()
            st.header("Analysis results")

            original_col, annotated_col = st.columns(2)

            with original_col:
                st.subheader("Original image")
                st.image(
                    analysis["original"],
                    use_container_width=True,
                )

            with annotated_col:
                st.subheader("AI annotations")
                st.image(
                    analysis["annotated"],
                    use_container_width=True,
                )

            detections = analysis["detections"]
            total = len(detections)

            st.subheader("Detection summary")

            first_metric, second_metric = st.columns(2)

            first_metric.metric("Predictions", total)
            second_metric.metric(
                "Damage category",
                "Pothole" if total else "None detected",
            )

            if not detections:
                st.info(
                    "The model did not detect potholes above "
                    "the selected confidence threshold."
                )
                st.warning(
                    "No detection does not mean the road is safe. "
                    "The model may miss damage."
                )

            st.divider()
            st.header("Review and verify")

            st.write(
                "Review each prediction and choose Pending, "
                "Accepted, or Rejected. Your decision records "
                "your review, not independently verified ground truth."
            )

            decisions = {}

            for detection in detections:
                detection_id = detection["id"]

                review_key = (
                    f"review_{analysis['image_hash']}_{detection_id}"
                )

                if review_key not in st.session_state:
                    st.session_state[review_key] = "Pending"

                with st.container(border=True):
                    st.write(
                        f"**Prediction {detection_id + 1}:** "
                        f"{detection['class']}"
                    )

                    st.write(
                        f"Model confidence: "
                        f"{detection['confidence']:.1%}"
                    )

                    st.caption(
                        f"Bounding box (x1, y1, x2, y2): "
                        f"{detection['box']}"
                    )

                    decision = st.radio(
                        "Your review decision",
                        options=[
                            "Pending",
                            "Accepted",
                            "Rejected",
                        ],
                        key=review_key,
                        horizontal=True,
                    )

                    decisions[detection_id] = decision

            # Count decisions AFTER the loop.
            accepted_count = sum(
                decision == "Accepted"
                for decision in decisions.values()
            )

            rejected_count = sum(
                decision == "Rejected"
                for decision in decisions.values()
            )

            pending_count = sum(
                decision == "Pending"
                for decision in decisions.values()
            )

            st.subheader("Verification summary")

            metric_a, metric_b, metric_c = st.columns(3)

            metric_a.metric("Accepted", accepted_count)
            metric_b.metric("Rejected", rejected_count)
            metric_c.metric("Pending", pending_count)

            st.caption(
                "Accepted and rejected indicate user decisions. "
                "Pending means the prediction has not been reviewed."
            )

            st.divider()
            st.subheader("Evidence details")

            st.write(
                f"**Report reference:** "
                f"SS-{analysis['image_hash'].upper()}"
            )

            st.write(
                f"**Analysis time (UTC):** "
                f"{analysis['timestamp']}"
            )

            st.write(
                f"**User-entered location:** "
                f"{analysis['location'] or 'Not provided'}"
            )

            st.warning(
                "StreetSight provides preliminary computer-vision "
                "evidence. Predictions can be inaccurate. Human "
                "review does not independently confirm the physical "
                "road condition."
            )

            
            st.subheader("Download evidence report")

            try:
                pdf_bytes = generate_evidence_pdf(
                    analysis,
                    decisions,
                )

                st.download_button(
                    label="Download PDF evidence report",
                    data=pdf_bytes,
                    file_name=(
                        f"StreetSight_{analysis['image_hash']}.pdf"
                    ),
                    mime="application/pdf",
                    type="primary",
                    use_container_width=True,
                )

            except Exception as error:
                st.error("Could not generate the PDF report.")
                st.caption(
                    f"Error: {type(error).__name__}: "
                    f"{str(error)[:250]}"
                )

st.divider()

st.caption(
    "StreetSight · Community road-damage documentation · "
    "No official government affiliation"
)
