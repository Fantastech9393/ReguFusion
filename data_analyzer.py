import pandas as pd
import numpy as np
import re
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()


def create_llm_with_fallback():
    """Initialize the OpenAI model with fallback to GPT-3.5-Turbo."""
    try:
        llm = ChatOpenAI(model="gpt-4o-2024-08-06", temperature=0)
        llm.invoke("ping")
        print("Using GPT-4o model for data analysis")
        return llm
    except Exception as e:
        print(f"GPT-4o unavailable or quota exceeded: {e}")
        print("Switching to GPT-3.5-Turbo fallback")
        return ChatOpenAI(model="gpt-3.5-turbo", temperature=0)


# --- Helper Functions ---------------------------------------------------------

def clean_column_name(col: str) -> str:
    """Convert technical column names to human-readable labels."""
    col = re.sub(r'(?<!^)(?=[A-Z])', ' ', col)
    col = col.replace('_', ' ')
    return col.strip().title()


def generate_summary(df: pd.DataFrame) -> str:
    """Return a readable, high-level summary of the dataset."""
    rows, cols = df.shape
    summary_lines = [f"The dataset contains {rows} records and {cols} columns."]

    # Identify data types
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    if num_cols:
        summary_lines.append(f"Numeric columns: {', '.join(num_cols)}.")
        summary_lines.append("Key statistics for numeric columns:")
        desc = df[num_cols].describe().round(2).to_dict()
        for col, stats in desc.items():
            summary_lines.append(f" - {clean_column_name(col)}: mean {stats['mean']}, min {stats['min']}, max {stats['max']}")
    if cat_cols:
        summary_lines.append(f"Categorical columns: {', '.join(cat_cols)}.")

    # Highlight missing data
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    if not missing_cols.empty:
        summary_lines.append("Columns with missing values:")
        for col, val in missing_cols.items():
            summary_lines.append(f" - {clean_column_name(col)}: {val} missing")

    return "\n".join(summary_lines)


def analyze_question(df: pd.DataFrame, question: str) -> str:
    """
    Use the LLM to reason about a user's question and generate
    a direct answer using both natural-language understanding and data context.
    """
    llm = create_llm_with_fallback()

    # Convert column names for clarity
    df_for_prompt = df.copy()
    df_for_prompt.columns = [clean_column_name(c) for c in df.columns]

    # Limit size of prompt to avoid token overload
    data_preview = df_for_prompt.head(10).to_dict(orient="records")

    prompt = (
        "You are a data analyst AI assistant. The user is asking a question "
        "about this dataset. Use the data preview and your reasoning skills "
        "to give a direct, concise answer. Do not provide code examples—just answer.\n\n"
        f"Dataset sample:\n{data_preview}\n\n"
        f"Question: {question}\n\n"
        "Answer clearly and conversationally:"
    )

    try:
        response = llm.invoke(prompt)
        if hasattr(response, "content"):
            return response.content.strip()
        elif isinstance(response, str):
            return response.strip()
        else:
            return "I couldn't generate a valid answer from the model."
    except Exception as e:
        return f"Error analyzing question: {e}"


# --- Entry Point (for direct testing) -----------------------------------------

if __name__ == "__main__":
    # Quick local test
    test_data = pd.DataFrame({
        "Employee": ["Alice", "Bob", "Charlie"],
        "Department": ["HR", "IT", "Finance"],
        "Salary": [60000, 75000, 68000],
        "Satisfaction": [8, 3, 6],
        "YearsAtCompany": [2, 5, 3],
        "PerformanceScore": [4, 2, 3]
    })

    print("=== SUMMARY ===")
    print(generate_summary(test_data))
    print("\n=== QUESTION TEST ===")
    print(analyze_question(test_data, "Who has the highest satisfaction score?"))

# --- Backward Compatibility for app.py ---------------------------------------

def analyze_data(df_or_path):
    """
    Compatibility wrapper so app.py can import analyze_data()
    regardless of previous function name changes.
    """
    if isinstance(df_or_path, pd.DataFrame):
        return generate_summary(df_or_path)
    elif isinstance(df_or_path, str):
        try:
            df = pd.read_csv(df_or_path)
            return generate_summary(df)
        except Exception as e:
            return f"Error reading CSV: {e}"
    else:
        return "Invalid input format for analyze_data(). Expected DataFrame or file path."