import re
import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Samsung Review Sentiment Analysis", page_icon="💬", layout="wide")

MODEL_PATH = Path(__file__).parent / "sentiment_model.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>|https?://\S+|www\.\S+", " ", text)
    text = text.replace("can't", "can not").replace("n't", " not")
    text = re.sub(r"[^a-z\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def sentiment_label(text, model):
    cleaned = clean_text(text)
    if not cleaned:
        return "Please enter review text."
    return model.predict([cleaned])[0]

st.title("💬 Samsung Customer Review Sentiment Analysis")
st.write("Classify customer reviews into **Positive**, **Neutral**, or **Negative** using a TF-IDF word + character n-gram model with a class-balanced Linear SVM.")

model = load_model()
tab1, tab2, tab3 = st.tabs(["Single Review", "Batch CSV Prediction", "About the Model"])

with tab1:
    review_title = st.text_input("Review title (optional)", placeholder="Example: Average phone")
    review_body = st.text_area("Enter customer review", height=160, placeholder="Write or paste a customer review...")
    if st.button("Predict Sentiment", type="primary"):
        review = (review_title + " " + review_body).strip()
        if not review:
            st.warning("Please enter a review before predicting.")
        else:
            prediction = sentiment_label(review, model)
            if prediction == "Positive":
                st.success(f"Predicted sentiment: {prediction} 😊")
            elif prediction == "Negative":
                st.error(f"Predicted sentiment: {prediction} ☹️")
            else:
                st.info(f"Predicted sentiment: {prediction} 😐")

with tab2:
    st.write("Upload a CSV containing a `review` column, or both `title` and `body` columns.")
    uploaded = st.file_uploader("Choose CSV file", type=["csv"])
    if uploaded is not None:
        batch = pd.read_csv(uploaded)
        if "review" in batch.columns:
            texts = batch["review"].fillna("").astype(str)
        elif {"title", "body"}.issubset(batch.columns):
            texts = batch["title"].fillna("").astype(str) + " " + batch["body"].fillna("").astype(str)
        else:
            st.error("CSV must contain `review`, or both `title` and `body` columns.")
            texts = None
        if texts is not None:
            cleaned = texts.map(clean_text)
            valid = cleaned.str.len() > 0
            predictions = [""] * len(batch)
            if valid.any():
                predicted = model.predict(cleaned[valid].tolist())
                for idx, label in zip(batch.index[valid], predicted):
                    predictions[batch.index.get_loc(idx)] = label
            batch["predicted_sentiment"] = predictions
            st.dataframe(batch, use_container_width=True)
            st.download_button(
                "Download predictions",
                batch.to_csv(index=False).encode("utf-8"),
                file_name="sentiment_predictions.csv",
                mime="text/csv"
            )

with tab3:
    st.subheader("Model details")
    st.markdown("""
    - **Text features:** Word-level TF-IDF (unigrams + bigrams) and character-level TF-IDF (3–5 character n-grams).
    - **Classifier:** Linear Support Vector Classifier (LinearSVC).
    - **Class handling:** `class_weight='balanced'` gives additional influence to less frequent sentiment classes, including Neutral.
    - **Labels:** Positive (ratings 4–5), Neutral (rating 3), Negative (ratings 1–2).
    """)
    st.warning("The model is a learning/demo project, not a substitute for human review. Predictions may be uncertain, especially for short or mixed-opinion text.")
