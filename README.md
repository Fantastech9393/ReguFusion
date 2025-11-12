ReguFusion

AI-Powered Data Intelligence Dashboard

Overview

ReguFusion is an interactive AI-driven analytics app that transforms CSV datasets into actionable insights.
It combines data visualization, automated exploratory analysis, and natural-language querying so users can understand their data without writing a single line of code.

Key Features

CSV Upload & Parsing
Upload any structured CSV file and automatically clean, format, and analyze its contents.

Exploratory Data Analysis (EDA)
Automatic summaries of numeric and categorical columns, missing values, and statistical distributions.

Dynamic Visualizations
Generate correlation heatmaps, histograms, boxplots, and scatter plots directly in the interface.

AI-Generated Insights
Built-in GPT-4o reasoning produces human-readable summaries describing dataset trends and patterns.

Conversational Q&A
Ask natural-language questions such as “Who has the highest satisfaction score?” or “Which department has the most employees?”
The AI interprets the dataset and returns direct answers, not code.

Error Handling & Model Fallback
Automatically switches to GPT-3.5-Turbo if GPT-4o is unavailable.

Tech Stack

Frontend / Framework: Streamlit

Language: Python

Libraries: Pandas, Seaborn, Matplotlib, Regex, Dotenv

LLM Integration: LangChain with OpenAI GPT-4o and GPT-3.5 fallback

Environment Management: Anaconda / virtualenv

Installation

Clone the repository

git clone https://github.com/yourusername/ReguFusion.git
cd ReguFusion


Create and activate a virtual environment

conda create -n regufusion python=3.10
conda activate regufusion


Install dependencies

pip install -r requirements.txt


Add your OpenAI API key to a .env file in the project root:

OPENAI_API_KEY=your_api_key_here

Usage

Run the Streamlit app:

streamlit run app.py


Then open the provided local URL in your browser.
Upload a CSV file to explore, visualize, and query your dataset.

Example Questions

Which employee has the highest satisfaction score?

What is the average salary by department?

Show a chart of performance versus years at company.

Which department has the most missing data?

Project Structure
ReguFusion/
│
├── app.py                # Main Streamlit application
├── analyzer.py           # Data summarization and AI reasoning
├── chatbot.py            # Chat interaction logic
├── data_analyzer.py      # Statistical analysis and insight generation
├── knowledge_agent.py    # Model creation and fallback logic
├── utils.py              # Helper functions and utilities
├── requirements.txt      # Project dependencies
└── data/                 # Example datasets

Example Output

User: Who has the highest salary?
AI: Kevin has the highest salary, which is $105,000.

User: Who has the lowest satisfaction score?
AI: Linda has the lowest satisfaction score, which is 1.


