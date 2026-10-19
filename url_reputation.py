
import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GOOGLE_SAFE_BROWSING_API_KEY")

def check_url_reputation(url):
    """Check whether Google Safe Browsing flags a URL."""

    if not API_KEY:
        return {
            "status": "unavailable",
            "message": "Safe Browsing API key is not configured."
        }

    endpoint = (
        "https://safebrowsing.googleapis.com/v4/threatMatches:find"
    )

    payload = {
        "client": {
            "clientId": "phishing-url-detector",
            "clientVersion": "1.0"
        },
        "threatInfo": {
            "threatTypes": [
                "MALWARE",
                "SOCIAL_ENGINEERING",
                "UNWANTED_SOFTWARE",
                "POTENTIALLY_HARMFUL_APPLICATION"
            ],
            "platformTypes": ["ANY_PLATFORM"],
            "threatEntryTypes": ["URL"],
            "threatEntries": [{"url": url}]
        }
    }

    try:
        response = requests.post(
            endpoint,
            params={"key": API_KEY},
            json=payload,
            timeout=10
        )

        if response.status_code != 200:
            return {
                "status": "unavailable",
                "message": (
                    f"Safe Browsing API error: HTTP "
                    f"{response.status_code}"
                )
            }

        data = response.json()

        if data.get("matches"):
            return {
                "status": "unsafe",
                "message": "Google Safe Browsing flagged this URL."
            }

        return {
            "status": "no_match",
            "message": (
                "No threat match found. This does not guarantee "
                "that the URL is safe."
            )
        }

    except requests.RequestException:
        return {
            "status": "unavailable",
            "message": "Could not reach Google Safe Browsing."
        }
