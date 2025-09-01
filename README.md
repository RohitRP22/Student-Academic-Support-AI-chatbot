🎓 Student RAG Chatbot (LangGraph + Groq)

This project is an AI-powered student performance assistant built using LangGraph, LangChain, and Groq’s LLaMA-3 model.
It integrates a Retrieval-Augmented Generation (RAG) system to answer queries about student performance, assignments, marks, and attendance.

🚀 Features

🔍 Student RAG Helper – retrieves academic data from a structured dataset.

🧠 LLM-Powered Chatbot – uses Groq’s LLaMA-3 (llama3-8b-8192) for intelligent responses.

⚡ LangGraph Workflow – tool-augmented conversation flow with conditional routing.

🛠️ Custom Tool Integration – structured RAG tool (student_progress_lookup) for academic queries.

💬 Interactive CLI Chat – ask natural language questions like:

"What is my overall progress in Data Structures?"

"How many assignments are pending for Ganesh?"

"Who are the top 3 students in AI?"

🗂️ Project Structure
├── helpers.py             # Contains StudentRAGHelper class (retrieval logic)
├── main.py                # Chatbot implementation (LangGraph + Groq + Tools)
├── .env                   # Store your GROQ_API_KEY
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation

🔑 Prerequisites

Python 3.9+

A valid Groq API key

⚙️ Installation

Clone the repo

git clone https://github.com/your-username/student-rag-chatbot.git
cd student-rag-chatbot


Create a virtual environment

python -m venv venv
source venv/bin/activate   # (Linux/Mac)
venv\Scripts\activate      # (Windows)


Install dependencies

pip install -r requirements.txt


Set up environment variables
Create a .env file in the root directory and add:

GROQ_API_KEY=your_groq_api_key_here

▶️ Usage

Run the chatbot:

python main.py


Example conversation:

💬 Student RAG Chatbot (LangGraph + Groq)
Type 'exit' to quit.

Examples: 
 - What is my overall progress in Data Structures?
 - How many assignments are pending for Ganesh?
 - Who are the top 3 students in AI?

You: How many assignments are pending for Ganesh?
Bot: Ganesh has 2 pending assignments in Machine Learning.

🛠️ Tech Stack

LangChain
 – LLM orchestration

LangGraph
 – stateful graph-based agent framework

Groq
 – fast LLM inference (LLaMA-3 model)

Pandas – dataset handling

dotenv – environment variable management

📌 Next Steps

✅ Add support for multiple datasets (marks, attendance, assignments).

✅ Extend RAG retrieval to include semantic search on notes.

🚀 Deploy chatbot as a web app (Streamlit/Flask/FastAPI).

🔒 Add authentication for multi-user access.

🤝 Contributing

Contributions are welcome! Feel free to fork this repo, create a feature branch, and submit a PR.