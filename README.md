Introduction

This project is a Streamlit-based chatbot that integrates Ollama for chat responses, ChromaDB for document storage and retrieval, and DuckDuckGo for web-enhanced search capabilities. It also provides functionalities like document management, word cloud generation, and knowledge graph visualization.

Features

Chat with Ollama AI, enhanced by web search and document retrieval.

Secure authentication with username and password.

Upload and store PDF and TXT documents for retrieval.

Query stored documents using semantic search.

Generate a word cloud from stored documents.

Generate a knowledge graph from stored documents.

Installation and Setup

Prerequisites

Ensure you have the following installed:

Python 3.x

Required dependencies (install using the command below)

Install Dependencies

Run the following command to install necessary dependencies:

pip install streamlit chromadb ollama sentence-transformers matplotlib plotly networkx wordcloud duckduckgo-search pymupdf

Running the Application

To start the application, use:

streamlit run your_script.py

Replace your_script.py with the filename containing the code.

Usage

Authentication

Enter your username and password.

Click the "Login" button.

If authenticated, you can access all functionalities.

Chat with Ollama

Type your question in the chat input field.

The chatbot fetches relevant documents from ChromaDB and web results from DuckDuckGo.

Ollama generates a response based on the retrieved context.

The response is displayed with the processing duration.

Document Management

Upload Documents: Upload a PDF or TXT file, enter a document ID, and click "Process Uploaded File."

Manually Add Documents: Enter document content and an ID, then click "Add Document."

View Stored Documents: Click "Show Documents" to list stored files.

Delete All Documents: Click "Delete All Documents" to clear the database.

Visualizations

Generate Word Cloud: Click "Generate Word Cloud" to create a word cloud from stored documents.

Generate Knowledge Graph: Click "Generate Knowledge Graph" to visualize keyword connections.

Notes

Ensure st.secrets contains user authentication details.

The chatbot filters inappropriate language using a predefined list.

The script uses ChromaDB to persist uploaded documents for future queries.
