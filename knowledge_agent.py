import os
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.tools import tool
from openai import OpenAIError

# Load environment variables
load_dotenv()

# LLM with fallback
def create_llm_with_fallback():
    """Try GPT-4o first, then fallback to GPT-3.5-turbo if unavailable or quota exceeded."""
    try:
        llm = ChatOpenAI(model="gpt-4o-2024-08-06", temperature=0)
        llm.invoke("ping")
        print("Using GPT-4o model")
        return llm
    except OpenAIError as e:
        print(f"GPT-4o unavailable or quota exceeded: {e}")
        print("Switching to GPT-3.5-turbo fallback")
        return ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    except Exception as e:
        print(f"Unexpected error initializing GPT-4o: {e}")
        print("Switching to GPT-3.5-turbo fallback")
        return ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

llm = create_llm_with_fallback()

# Define a simple Pandas tool
@tool
def describe_data():
    """Describe key statistics from the HR dataset."""
    data = {
        "Employee": ["Alice", "Bob", "Charlie"],
        "Satisfaction": [8, 3, 6]
    }
    df = pd.DataFrame(data)
    return {
        "summary": f"Dataset loaded with {len(df)} records.",
        "lowest": f"Lowest satisfaction: {df.loc[df['Satisfaction'].idxmin(), 'Employee']}"
    }

# Create the agent using the LangChain v1 factory
agent = create_agent(model=llm, tools=[describe_data])

# Run the agent
input_payload = {
    "messages": [{"role": "user", "content": "Analyze the employee satisfaction data."}]
}

try:
    response = agent.invoke(input_payload)
    print("Agent response:", response)
    print("Knowledge agent test completed successfully.")
except Exception as e:
    print("Agent execution failed:", str(e))
