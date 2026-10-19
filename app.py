
from pathlib import Path
from urllib.parse import urlparse

import joblib
import pandas as pd
import streamlit as st

from feature_extraction import extract_host_features
from url_reputation import check_url_reputation


# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Intelligent Phishing URL Detector",
    page_icon="🛡️",
    layout="centered",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "phishing_model_v4.joblib"


# --------------------------------------------------
# 2. Load trained model
# --------------------------------------------------

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH.name}. "
            "Make sure the model file is in the project folder."
        )

    return joblib.load(MODEL_PATH)


# --------------------------------------------------
# 3. URL validation
# --------------------------------------------------

def validate_url(url):
    """Validate URL format without visiting the website."""
    try:
        parsed = urlparse(url)

        if parsed.scheme not in ("http", "https"):
            return False

        if not parsed.hostname:
            return False

        # Check for malformed ports.
        _ = parsed.port

        return True

    except (ValueError, TypeError):
        return False


# --------------------------------------------------
# 4. Application interface
# --------------------------------------------------

st.title("🛡️ Intelligent Phishing URL Detection")

st.write(
    "Analyze URLs using a machine-learning model and "
    "Google Safe Browsing threat intelligence."
)

st.info(
    "This application does not open the submitted website. "
    "Machine-learning predictions are estimates, and a clean "
    "reputation lookup does not guarantee safety."
)


# --------------------------------------------------
# 5. Load model
# --------------------------------------------------

try:
    model = load_model()
    st.success("Machine-learning model loaded successfully.")

except Exception as exc:
    st.error(f"Unable to load the model: {exc}")
    st.stop()


# --------------------------------------------------
# 6. URL input
# --------------------------------------------------

url = st.text_input(
    "Enter the complete URL",
    placeholder="https://example.com",
    help="Enter a complete URL beginning with http:// or https://.",
)

check_button = st.button(
    "🔍 Analyze URL",
    type="primary",
    use_container_width=True,
)


# --------------------------------------------------
# 7. Machine-learning and reputation analysis
# --------------------------------------------------

if check_button:
    cleaned_url = url.strip()

    if not cleaned_url:
        st.warning("Please enter a URL.")

    elif not validate_url(cleaned_url):
        st.warning(
            "Invalid URL. Enter a complete HTTP or HTTPS URL "
            "with a valid hostname and valid port, if specified."
        )

    else:
        try:
            # Extract features using the shared function.
            features = extract_host_features(cleaned_url)

            # Build the input expected by the trained pipeline.
            model_input = pd.DataFrame([
                {
                    "URL": cleaned_url,
                    **features,
                }
            ])

            # Run machine-learning prediction.
            prediction = int(model.predict(model_input)[0])
            probabilities = model.predict_proba(model_input)[0]
            classes = list(model.classes_)

            if 0 not in classes or 1 not in classes:
                raise ValueError(
                    "Unexpected model classes. Expected "
                    "0 = phishing and 1 = legitimate."
                )

            phishing_probability = float(
                probabilities[classes.index(0)]
            )

            legitimate_probability = float(
                probabilities[classes.index(1)]
            )

            # ----------------------------------------------
            # 7A. Machine-learning result
            # ----------------------------------------------

            st.divider()
            st.subheader("1. Machine-Learning Analysis")

            st.metric(
                label="Estimated phishing probability",
                value=f"{phishing_probability:.2%}",
            )

            st.progress(phishing_probability)

            if prediction == 0:
                st.error("⚠️ ML model: Potential phishing")

                st.write(
                    "The model classified this URL as phishing. "
                    "Avoid entering sensitive information until "
                    "you have independently verified the URL."
                )

            elif prediction == 1:
                st.success("ML model: Classified as legitimate")

                st.write(
                    "The model classified this URL as legitimate. "
                    "This does not guarantee that the website "
                    "or its content is safe."
                )

            st.caption(
                "Estimated legitimate probability: "
                f"{legitimate_probability:.2%}"
            )

            # ----------------------------------------------
            # 7B. Extracted features
            # ----------------------------------------------

            with st.expander("View extracted URL features"):
                st.dataframe(
                    pd.DataFrame(
                        features.items(),
                        columns=["Feature", "Value"],
                    ),
                    hide_index=True,
                    use_container_width=True,
                )

            # ----------------------------------------------
            # 7C. Google Safe Browsing check
            # ----------------------------------------------

            st.divider()
            st.subheader("2. Google Safe Browsing Check")

            try:
                with st.spinner(
                    "Checking URL reputation with Google Safe Browsing..."
                ):
                    reputation = check_url_reputation(cleaned_url)

                if not isinstance(reputation, dict):
                    reputation = {
                        "status": "unavailable",
                        "message": "The reputation service returned "
                                   "an unexpected response.",
                    }

            except Exception:
                reputation = {
                    "status": "unavailable",
                    "message": "The reputation check could not be "
                               "completed. Check the API configuration "
                               "and network connection.",
                }

            reputation_status = reputation.get(
                "status", "unavailable"
            )

            if reputation_status == "unsafe":
                st.error(
                    "🚨 Google Safe Browsing reported a known threat."
                )

                st.warning(
                    "Treat this URL as unsafe. Do not enter passwords, "
                    "payment details, or other sensitive information."
                )

            elif reputation_status == "no_match":
                st.info(
                    "No known threat match was found by Google "
                    "Safe Browsing. This does not prove the URL is safe."
                )

            else:
                st.warning(
                    "Google Safe Browsing could not complete the check. "
                    "The ML result remains available independently."
                )

            st.caption(
                str(
                    reputation.get(
                        "message",
                        "No additional reputation details are available.",
                    )
                )
            )

            # ----------------------------------------------
            # 7D. Combined assessment
            # ----------------------------------------------

            st.divider()
            st.subheader("3. Combined Assessment")

            if reputation_status == "unsafe":
                st.error(
                    "HIGH CONCERN: Google Safe Browsing reported "
                    "a known threat. Treat the URL as unsafe."
                )

            elif (
                reputation_status == "no_match"
                and prediction == 0
            ):
                st.warning(
                    "SUSPICIOUS: The ML model flagged this URL, "
                    "although Safe Browsing found no known threat match. "
                    "Further verification is recommended."
                )

            elif (
                reputation_status == "no_match"
                and prediction == 1
            ):
                st.info(
                    "NO THREAT DETECTED BY THESE CHECKS: Neither "
                    "source flagged the URL. This is not a guarantee "
                    "that the URL is safe."
                )

            else:
                st.warning(
                    "INCOMPLETE ASSESSMENT: The reputation check "
                    "was unavailable. Review the ML prediction "
                    "without treating it as a definitive verdict."
                )

        except Exception as exc:
            st.error(f"URL analysis failed: {exc}")


# --------------------------------------------------
# 8. Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Academic project • PhiUSIIL dataset and additional URL examples "
    "• Logistic Regression pipeline • Google Safe Browsing lookup. "
    "Both methods can miss threats or produce incorrect assessments. "
    "Always verify suspicious URLs independently."
)
