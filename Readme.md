# Assignment: Personalized SmartShopper Assistant

## Stack:
1. Runtime: Python 3.10.
2. Database: MongoDB.
3. Embedding model: all-mpnet-base-v2.
4. LLM model: gpt-4.1.
5. Interface: Streamlit.

## Prerequisites:
1. MongoDB connection string from: https://cloud.mongodb.com/
2. OpenAI API key from: https://platform.openai.com/

## Setup:
1. `git clone https://github.com/RifqiAnshariR/agentic-rag-haystack.git`
2. `cd agentic-rag-haystack`
3. `py -3.10 -m venv .venv` and activate it `.venv\Scripts\activate`
4. `pip install -r requirements.txt`
5. Make .env file contains: MONGO_CONNECTION_STRING and OPENAI_API_KEY.

## How to run:
1. To run ingestor: `python ingestor.py`
2. To run Streamlit: `streamlit run app.py`