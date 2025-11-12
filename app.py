import os
import tempfile
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import re

# Local imports
from knowledge_agent import create_llm_with_fallback
from data_analyzer import analyze_data

# Load environment variables
load_dotenv()

# Initialize LLM + session memory
llm = create_llm_with_fallback()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "df" not in st.session_state:
    st.session_state.df = None

# -----------------------------------------------------------
# Streamlit UI Configuration
# -----------------------------------------------------------
st.set_page_config(
    page_title="ReguFusion - Data Intelligence App",
    layout="wide"
)
st.title("ReguFusion Data Intelligence Dashboard")
st.caption("Upload your dataset and let AI analyze, visualize, and explain it.")

# ===========================================================
# 1. File Upload & Base Handling
# ===========================================================
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    # --- FIXED UPLOAD BLOCK (handles empty or unreadable temp files) ---
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        file_path = tmp_file.name

    try:
        # Attempt reading from the temporary file
        df = pd.read_csv(file_path)
        if df.empty or df.columns.size == 0:
            # Fallback: read directly from memory
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        st.stop()

    # Fix camelCase and underscores in column names
    df.columns = [
        re.sub(r'(?<!^)(?=[A-Z])', ' ', col).replace("_", " ").title()
        for col in df.columns
    ]

    st.session_state.df = df
    st.success("File uploaded and loaded successfully.")

    # =======================================================
    # Dataset Preview
    # =======================================================
    st.subheader("Dataset Preview")

    # If dataset is small (<= 50 rows), show all; else show first 20
    if len(df) <= 50:
        st.dataframe(df)
    else:
        st.dataframe(df.head(20))
        with st.expander("Show full dataset"):
            st.dataframe(df)


    # =======================================================
    # 2. Basic Data Overview
    # =======================================================
    st.subheader("Dataset Overview")
    st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")

    col1, col2 = st.columns(2)
    with col1:
        st.write("Column Data Types:")
        st.dataframe(df.dtypes.rename("Data Type"))

    with col2:
        st.write("Missing Values:")
        st.dataframe(df.isnull().sum().rename("Missing Count"))

    # =======================================================
    # 3. Exploratory Data Analysis (EDA)
    # =======================================================
    st.subheader("Quick Stats (EDA Summary)")

    numeric_cols = df.select_dtypes(include=["number"]).columns
    if not numeric_cols.empty:
        st.dataframe(df[numeric_cols].describe().T)
    else:
        st.warning("No numeric columns found for EDA summary.")

    # =======================================================
    # 4. Visualizations
    # =======================================================
    st.subheader("Visualizations")

    if not numeric_cols.empty:
        chart_option = st.selectbox(
            "Choose a visualization type:",
            ["Correlation Heatmap", "Histogram", "Boxplot", "Scatter Plot"]
        )

        if chart_option == "Correlation Heatmap":
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

        elif chart_option == "Histogram":
            selected_col = st.selectbox("Select a numeric column:", numeric_cols)
            fig, ax = plt.subplots()
            sns.histplot(df[selected_col], kde=True, ax=ax)
            ax.set_title(f"Distribution of {selected_col}")
            st.pyplot(fig)

        elif chart_option == "Boxplot":
            selected_col = st.selectbox("Select a numeric column:", numeric_cols)
            fig, ax = plt.subplots()
            sns.boxplot(x=df[selected_col], ax=ax)
            ax.set_title(f"Boxplot of {selected_col}")
            st.pyplot(fig)

        elif chart_option == "Scatter Plot":
            x_col = st.selectbox("X-Axis", numeric_cols, key="xaxis")
            y_col = st.selectbox("Y-Axis", numeric_cols, key="yaxis")
            fig, ax = plt.subplots()
            sns.scatterplot(x=df[x_col], y=df[y_col], ax=ax)
            ax.set_title(f"{y_col} vs {x_col}")
            st.pyplot(fig)
    else:
        st.warning("No numeric columns available for visualization.")

    # =======================================================
    # 5. AI-Powered Analysis Summary
    # =======================================================
    st.subheader("AI Summary Insights")

    if st.button("Generate AI Analysis Summary"):
        with st.spinner("AI is analyzing your dataset..."):
            try:
                summary = analyze_data(file_path)
                st.write(summary)
            except Exception as e:
                st.error(f"Error during AI analysis: {e}")

    # =======================================================
    # 6. Interactive Q&A Chat about Dataset
    # =======================================================
    st.subheader("Ask AI about your data")

    user_query = st.text_input("Ask a question about your dataset:", placeholder="Type here...")

    if user_query:
        with st.spinner("AI thinking..."):
            try:
                # Limit dataset size for token safety (first 50 rows)
                data_preview = df.head(50).to_dict(orient="records")

                # Build a reasoning prompt with actual dataset content
                prompt = (
                    "You are a professional data analyst. "
                    "Use the dataset values provided below to directly answer the user's question. "
                    "Do NOT show code or describe how to compute; just give the actual answer. "
                    "If the dataset clearly answers the question, state the result clearly. "
                    "If not enough data is available, say so.\n\n"
                    f"Dataset preview:\n{data_preview}\n\n"
                    f"User question: {user_query}\n\n"
                    "Answer in one or two clear sentences:"
                )

                response = llm.invoke(prompt)
                answer = getattr(response, "content", str(response))

                # Save to chat history
                st.session_state.chat_history.append(("You", user_query))
                st.session_state.chat_history.append(("AI", answer))
            except Exception as e:
                st.session_state.chat_history.append(("AI", f"Error: {str(e)}"))

    if st.session_state.chat_history:
        for sender, msg in st.session_state.chat_history[-10:]:
            st.markdown(f"**{sender}:** {msg}")

else:
    st.info("Upload a CSV file to begin analysis.")