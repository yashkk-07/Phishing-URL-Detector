
import ipaddress
from pathlib import Path
from urllib.parse import urlparse

import joblib
import pandas as pd
import streamlit as st


# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Intelligent Phishing URL Detector",
    page_icon="🛡️",
    layout="centered",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "phishing_model_v2.joblib"


# --------------------------------------------------
# 2. Feature extraction
# --------------------------------------------------

def extract_host_features(url):
    """
    Extract the eight structural features expected by
    the saved model.

    Important: verify these definitions against the
    original training function before production use.
    """

    url = str(url).strip()
    parsed = urlparse(url if "://" in url else "//" + url)

    hostname = (parsed.hostname or "").lower()
    host_parts = [part for part in hostname.split(".") if part]

    try:
        ipaddress.ip_address(hostname)
        host_is_ip = 1
    except ValueError:
        host_is_ip = 0

    return {
        "HostLength": len(hostname),
        "HostIsIP": host_is_ip,
        "SubdomainCount": max(len(host_parts) - 2, 0),
        "HasHyphenInHost": int("-" in hostname),
        "HostDigitCount": sum(char.isdigit() for char in hostname),
        "PathLength": len(parsed.path),
        "QueryLength": len(parsed.query),
        "URLLength": len(url),
    }


# --------------------------------------------------
# 3. Load the trained model
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# --------------------------------------------------
# 4. Application interface
# --------------------------------------------------

st.title("🛡️ Intelligent Phishing URL Detection")

st.write(
    "Analyze URL characteristics using a machine-learning "
    "model trained on the PhiUSIIL dataset."
)

st.info(
    "This tool analyzes URL text only. It does not visit "
    "the website. Predictions are estimates, not guarantees "
    "of safety."
)

try:
    model = load_model()
    st.success("Detection model loaded successfully.")
except Exception as exc:
    st.error(f"Unable to load the model: {exc}")
    st.stop()


# --------------------------------------------------
# 5. URL input and prediction
# --------------------------------------------------

url = st.text_input(
    "Enter the complete URL",
    placeholder="https://example.com",
    help="Include http:// or https:// in the URL.",
)

if st.button("🔍 Check URL", type="primary", use_container_width=True):

    cleaned_url = url.strip()

    if not cleaned_url:
        st.warning("Please enter a URL.")

    elif not cleaned_url.lower().startswith(("http://", "https://")):
        st.warning("Enter a complete URL starting with http:// or https://.")

    else:
        try:
            parsed = urlparse(cleaned_url)

            if not parsed.hostname:
                st.warning("Please enter a valid URL containing a hostname.")
                st.stop()

            # Create all nine input columns required by the model
            features = extract_host_features(cleaned_url)

            model_input = pd.DataFrame([
                {
                    "URL": cleaned_url,
                    **features,
                }
            ])

            # Predict class and probabilities
            prediction = int(model.predict(model_input)[0])
            probabilities = model.predict_proba(model_input)[0]
            classes = list(model.classes_)

            phishing_index = classes.index(0)
            legitimate_index = classes.index(1)

            phishing_probability = float(
                probabilities[phishing_index]
            )
            legitimate_probability = float(
                probabilities[legitimate_index]
            )

            # Display result
            st.subheader("Analysis Result")

            st.metric(
                label="Model-estimated phishing probability",
                value=f"{phishing_probability:.2%}",
            )

            st.progress(phishing_probability)

            if prediction == 0:
                st.error("⚠️ Potential phishing URL")

                st.write(
                    "The model classified this URL as phishing. "
                    "Avoid entering passwords, payment details, "
                    "or other sensitive information."
                )

            elif prediction == 1:
                st.success("✅ Model prediction: Legitimate")

                st.write(
                    "The model classified this URL as legitimate. "
                    "This does not establish that the website is safe."
                )

            st.caption(
                f"Model-estimated legitimate probability: "
                f"{legitimate_probability:.2%}"
            )

            with st.expander("View extracted URL features"):
                st.dataframe(
                    pd.DataFrame(
                        features.items(),
                        columns=["Feature", "Value"],
                    ),
                    hide_index=True,
                    use_container_width=True,
                )

        except Exception as exc:
            st.error(f"URL analysis failed: {exc}")


# --------------------------------------------------
# 6. Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Academic project • PhiUSIIL dataset • Machine-learning-based "
    "URL analysis. Predictions may be incorrect. Verify suspicious "
    "URLs independently before interacting with them."
)
