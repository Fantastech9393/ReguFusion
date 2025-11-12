import os
import json
import sqlite3
from datetime import datetime
import pandas as pd


# ---------- SQLite Setup ----------
DB_PATH = os.path.join("database", "regufusion.db")

def init_db():
    """Create database and required tables if they don't exist."""
    os.makedirs("database", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS chat_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            user_input TEXT,
            model_response TEXT,
            source TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            user_input TEXT,
            model_response TEXT,
            feedback TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            start_time TEXT,
            end_time TEXT,
            user_id TEXT
        )
    """)

    conn.commit()
    conn.close()


def log_chat(user_input, model_response, source="ChatAgent"):
    """Insert each chat message into the database."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO chat_logs (timestamp, user_input, model_response, source)
        VALUES (?, ?, ?, ?)
    """, (datetime.now().isoformat(), user_input, model_response, source))
    conn.commit()
    conn.close()


def log_feedback(user_input, model_response, feedback):
    """Store thumbs-up or thumbs-down feedback."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO feedback (timestamp, user_input, model_response, feedback)
        VALUES (?, ?, ?, ?)
    """, (datetime.now().isoformat(), user_input, model_response, feedback))
    conn.commit()
    conn.close()


def fetch_history(limit=25):
    """Return recent chat history as a DataFrame."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        f"SELECT * FROM chat_logs ORDER BY id DESC LIMIT {limit}", conn)
    conn.close()
    return df


# ---------- JSON Utility (kept for compatibility) ----------
def load_json(path):
    """Gracefully read JSON files; return empty list on error."""
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_json(data, path):
    """Safely overwrite JSON files."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# Initialize database when module loads
init_db()
