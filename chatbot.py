import os
import json
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from utils import log_chat

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ChatAgent:
    """Conversational agent for ReguFusion with logging and context memory."""

    def __init__(self, prompts_file="prompts.json", log_dir="logs"):
        self.prompts_file = prompts_file
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(
            log_dir,
            f"session_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        # Load few-shot examples
        if os.path.exists(prompts_file):
            with open(prompts_file, "r", encoding="utf-8") as f:
                self.prompts = json.load(f)
        else:
            self.prompts = []

        self.history = []

    def _save_trace(self, user_query, model_response):
        """Store JSON log for debugging alongside SQLite logs."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "model": "gpt-4o-mini",
            "user_query": user_query,
            "assistant_reply": model_response,
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def chat(self, query):
        """Run conversation with OpenAI model, store logs."""
        system_prompt = (
            "You are ReguFusion, an AI assistant that helps interpret compliance "
            "and regulatory data, summarizing results clearly and transparently."
        )

        messages = [{"role": "system", "content": system_prompt}]
        for p in self.prompts:
            messages.append({"role": "user", "content": p.get("input", "")})
            messages.append({"role": "assistant", "content": p.get("output", "")})
        messages.append({"role": "user", "content": query})

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # Stable and fast model
                messages=messages,
                temperature=0.6
            )
            reply = response.choices[0].message.content.strip()
            self._save_trace(query, reply)
            log_chat(query, reply, "ChatAgent")  # <-- now stored in SQLite
            return reply
        except Exception as e:
            error_msg = f"Error: {e}"
            log_chat(query, error_msg, "ChatAgent")
            return error_msg


# Quick test
if __name__ == "__main__":
    bot = ChatAgent()
    print(bot.chat("Summarize GDPR compliance principles."))
