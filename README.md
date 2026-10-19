# Intelligent Phishing URL Detection System

A machine-learning-based web application that analyzes URLs for potential phishing threats using Logistic Regression and Google Safe Browsing.

## Features

- Phishing URL classification using character-level TF-IDF and URL features.
- Estimated phishing probability.
- Google Safe Browsing reputation lookup.
- Combined assessment of ML predictions and reputation results.
- Interactive Streamlit web interface.

## Technology Stack

Python, Pandas, Scikit-learn, TF-IDF, Logistic Regression, Streamlit, Joblib and Google Safe Browsing API.

## Model Evaluation

The recorded V4 test split achieved **99.5607% accuracy** across 40,290 samples.

- Phishing precision: 99.98%
- Phishing recall: 99.15%
- Phishing F1-score: 99.56%

The `reports/` directory contains the confusion matrix and classification report.

These results represent performance on the selected test split and do not guarantee equivalent performance against real-world phishing URLs.

## Project Structure

- `app.py` — Streamlit application
- `feature_extraction.py` — URL feature extraction
- `url_reputation.py` — Google Safe Browsing integration
- `phishing_model_v4.joblib` — Trained model
- `train_model_v4.py` — Training and evaluation script
- `reports/` — Evaluation outputs
- `requirements.txt` — Python dependencies

## Run Locally

1. Clone the repository.
2. Create and activate a Python virtual environment.
3. Install dependencies using `pip install -r requirements.txt`.
4. Configure the Google Safe Browsing API key securely.
5. Run `python -m streamlit run app.py`.

Keep API keys in environment variables or deployment secrets. Never commit `.env` or API credentials.

## Limitations

The model can produce false positives and false negatives. During testing, Amazon India was incorrectly classified as phishing. A clean Google Safe Browsing result does not prove a URL is safe.

## Future Improvements

- Evaluate using an independent, recent dataset.
- Reduce false positives on legitimate websites.
- Improve model explainability and robustness.
- Expand testing against emerging phishing patterns.

## Disclaimer

Developed for educational and defensive cybersecurity purposes.
