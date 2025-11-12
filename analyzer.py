import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import joblib
import os
from datetime import datetime
from utils import log_chat


class DataAgent:
    """Analyzes numeric and categorical data and logs insights."""

    def load_data(self, uploaded_file):
        """Read CSV or Excel file and return DataFrame."""
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        return df

    def analyze(self, df):
        """Generate summary statistics and correlation plots."""
        summary = df.describe(include="all")
        figs = []

        numeric_cols = df.select_dtypes(include="number")
        if not numeric_cols.empty:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(numeric_cols.corr(), annot=True, cmap="coolwarm", ax=ax)
            ax.set_title("Correlation Heatmap")
            figs.append(fig)

        # Bar chart for top categories
        cat_cols = df.select_dtypes(include="object")
        for col in cat_cols.columns[:3]:
            fig, ax = plt.subplots(figsize=(6, 4))
            df[col].value_counts().head(10).plot(kind="bar", ax=ax)
            ax.set_title(f"Top Values in '{col}'")
            figs.append(fig)

        # Log summary
        log_chat("Dataset Analyzed", f"Summary: {summary.head().to_dict()}", "DataAgent")
        return summary, figs


class SentimentAgent:
    """Performs sentiment analysis with TextBlob and a demo classifier."""

    def __init__(self):
        self.model_path = "sentiment_model.pkl"
        self.model = None
        self.vectorizer = None
        if os.path.exists(self.model_path):
            self.model, self.vectorizer = joblib.load(self.model_path)

    def _train_mini_model(self):
        """Train and save a tiny logistic regression sentiment model."""
        sample = [
            ("I love this policy update, it’s fantastic!", 1),
            ("This process is confusing and frustrating.", 0),
            ("Our team collaboration has improved greatly.", 1),
            ("I am unhappy with the current workflow.", 0),
        ]
        texts, labels = zip(*sample)
        vectorizer = CountVectorizer(stop_words="english")
        X = vectorizer.fit_transform(texts)
        model = LogisticRegression()
        model.fit(X, labels)
        joblib.dump((model, vectorizer), self.model_path)
        self.model, self.vectorizer = model, vectorizer

    def analyze(self, df):
        """Create sentiment visuals and log results."""
        text_cols = df.select_dtypes(include="object")
        if text_cols.empty:
            return None, None

        if self.model is None:
            self._train_mini_model()

        all_text = " ".join(df[col].astype(str).sum() for col in text_cols.columns)
        sentiment_scores = [
            TextBlob(t).sentiment.polarity for t in all_text.split(".") if t
        ]
        avg_sentiment = np.mean(sentiment_scores)

        # Sentiment bar
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(["Average Sentiment"], [avg_sentiment],
                color="green" if avg_sentiment > 0 else "red")
        ax.set_xlim(-1, 1)
        ax.set_title("Sentiment Polarity (-1 negative → +1 positive)")

        # Word cloud
        wc = WordCloud(stopwords=STOPWORDS, background_color="white").generate(all_text)
        fig_wc, ax_wc = plt.subplots(figsize=(6, 4))
        ax_wc.imshow(wc, interpolation="bilinear")
        ax_wc.axis("off")
        ax_wc.set_title("Word Cloud")

        log_chat("Sentiment Analysis", f"Average sentiment: {avg_sentiment:.2f}", "SentimentAgent")
        return [fig, fig_wc]


# Quick test block
if __name__ == "__main__":
    agent = DataAgent()
    test_df = pd.DataFrame({
        "Satisfaction": [4, 5, 3, 2, 4],
        "Comments": [
            "Great place to work.",
            "Needs improvement.",
            "Happy overall.",
            "Too much paperwork.",
            "Excellent benefits!"
        ]
    })
    print("Running numeric + sentiment analysis test...")
    data_summary, _ = agent.analyze(test_df)
    sentiment_agent = SentimentAgent()
    figs = sentiment_agent.analyze(test_df)
    print("✅ Analyzer test completed successfully.")
