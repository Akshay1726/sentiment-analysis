# Samsung Customer Review Sentiment Analysis — Streamlit

## Files
- `app.py` — Streamlit web application (single-review and batch CSV prediction)
- `sentiment_model.joblib` — trained TF-IDF + class-balanced LinearSVC pipeline
- `requirements.txt` — Python dependencies

## Run locally
1. Install Python 3.10 or later.
2. Open a terminal in this project folder.
3. Create and activate a virtual environment (recommended).
4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Start the application:

   ```bash
   streamlit run app.py
   ```

6. Open the local URL printed by Streamlit (usually http://localhost:8501).

## Batch prediction
Upload a CSV containing either:
- a `review` column, or
- both `title` and `body` columns.

The app displays predictions and lets you download the result as a CSV.

## Model improvement
The training notebook compares a plain TF-IDF + LinearSVC baseline with a model using word and character n-grams and balanced class weights. The held-out test split used `random_state=42` and stratification. On this split, Neutral F1 improved from 0.12 (baseline) to 0.30 (balanced word+character model). Results depend on the dataset and split; no F1 score is guaranteed on new data.

## Deployment
For Streamlit Community Cloud:
1. Push `app.py`, `sentiment_model.joblib`, and `requirements.txt` to a GitHub repository.
2. Go to https://share.streamlit.io/ and sign in.
3. Select the repository, branch, and `app.py`.
4. Click Deploy.

Keep the model file in the repository alongside `app.py`. Do not upload private customer data or secrets.
