import streamlit as st
import chromadb
import time
import ollama
import fitz
from duckduckgo_search import DDGS
from sentence_transformers import SentenceTransformer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objects as go

chroma_client = chromadb.PersistentClient(path="./chroma_db")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
collection = chroma_client.get_or_create_collection(name="rag_collection")

SWEAR_WORDS = ["badword1", "badword2", "badword3"]

def contains_swear_words(text):
    return any(word in text.lower() for word in SWEAR_WORDS)

def add_document_to_collection(document, doc_id, metadata=None):
    embedding = embedding_model.encode([document])[0]
    collection.add(documents=[document], ids=[doc_id], embeddings=[embedding], metadatas=[metadata] if metadata else [{}])

def query_chromadb(query_text, n_results=1):
    query_embedding = embedding_model.encode([query_text])[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)
    
    documents = results.get("documents", [])
    
    flattened_docs = [" ".join(doc) if isinstance(doc, list) else str(doc) for doc in documents]
    
    return flattened_docs, results.get("metadatas", [])

def query_ollama(prompt):
    response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])
    return response.get("message", {}).get("content", "No response.")

def web_search(query):
    results = []
    with DDGS() as ddgs:
        search_results = ddgs.text(query, max_results=3)
        for result in search_results:
            results.append(result["body"])
    return results

def rag_pipeline(query_text):
    if contains_swear_words(query_text):
        return "Your query contains inappropriate language. Please rephrase your question."
    
    retrieved_docs, _ = query_chromadb(query_text)

    context = ""
    if retrieved_docs:
        context = "\n".join([str(doc) for doc in retrieved_docs])
    else:
        context = "No relevant documents found in memory."

    web_results = web_search(query_text)
    if web_results:
        context += "\n".join(web_results)

    augmented_prompt = f"Based on the latest information:\n{context}\n\nQuestion: {query_text}\nAnswer:"
    return query_ollama(augmented_prompt)

def extract_text_from_file(uploaded_file):
    if uploaded_file.name.endswith(".pdf"):
        pdf_reader = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        return " ".join([page.get_text() for page in pdf_reader])
    elif uploaded_file.name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8")
    else:
        st.error("Unsupported file format. Please upload a PDF or TXT file.")
        return None

def generate_word_cloud(text):
    if isinstance(text, list):  
        text = " ".join([" ".join(doc) if isinstance(doc, list) else str(doc) for doc in text])
    
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)


def generate_knowledge_graph(docs):
    G = nx.Graph()
    
    if isinstance(docs, list): 
        docs = [" ".join(doc) if isinstance(doc, list) else str(doc) for doc in docs]
    
    for doc in docs:
        words = doc.split()[:10]  
        for i in range(len(words) - 1):
            G.add_edge(words[i], words[i + 1])
    
    pos = nx.spring_layout(G)
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=list(G.nodes()),
        textposition="top center",
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=10,
            color=[],
            line_width=2))
    
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
    
    st.plotly_chart(fig)


def authenticate(username, password):
    users = st.secrets.get("users", {})
    return users.get(username) == password

def main():
    st.title("Chat with Web-Enhanced Ollama")
    
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if authenticate(username, password):
                st.session_state.authenticated = True
                st.success("Logged in successfully!")
            else:
                st.error("Invalid username or password.")
        return
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'short_term_memory' not in st.session_state:
        st.session_state.short_term_memory = []

    st.sidebar.header("Document Management")
    uploaded_file = st.sidebar.file_uploader("Upload PDF or TXT File", type=["pdf", "txt"])
    if st.sidebar.button("Process Uploaded File"):
        if uploaded_file:
            file_content = extract_text_from_file(uploaded_file)
            if file_content:
                doc_id = st.sidebar.text_input("Enter Document ID for Upload", key="upload_doc_id")
                if doc_id:
                    metadata = {"filename": uploaded_file.name, "timestamp": time.time()}
                    add_document_to_collection(file_content, doc_id, metadata)
                    st.sidebar.success("File processed and added successfully!")
                else:
                    st.sidebar.error("Please provide a document ID.")
        else:
            st.sidebar.error("Please upload a file.")
    
    new_doc = st.sidebar.text_area("New Document Content")
    doc_id = st.sidebar.text_input("New Document ID", key="new_doc_id")
    if st.sidebar.button("Add Document"):
        if new_doc and doc_id:
            add_document_to_collection(new_doc, doc_id)
            st.sidebar.success("Document added successfully!")
        else:
            st.sidebar.error("Please provide both document content and ID.")

    if st.sidebar.button("Show Documents"):
        try:
            docs, metadata = query_chromadb("", n_results=100)
            st.sidebar.write(f"Total documents in collection: {len(collection.get()['ids'])}")
            st.sidebar.write("Stored Documents:")
            for idx, doc in enumerate(docs):
                title = metadata[idx].get("filename", "Untitled") if metadata and isinstance(metadata[idx], dict) else "Untitled"
                st.sidebar.write(f"Title: {title}, Content: {doc[:100]}...") 
        except Exception as e:
            st.sidebar.error(f"Error retrieving documents: {e}")

    if st.sidebar.button("Delete All Documents"):
        all_doc_ids = collection.get()["ids"]
        if all_doc_ids:
            collection.delete(ids=all_doc_ids)
            st.sidebar.success("All documents deleted.")
        else:
            st.sidebar.info("No documents to delete.")
    
    if st.sidebar.button("Generate Word Cloud"):
        try:
            docs, _ = query_chromadb("", n_results=100)
            if docs:
                all_text = " ".join(docs)
                generate_word_cloud(all_text)
            else:
                st.sidebar.info("No documents available to generate word cloud.")
        except Exception as e:
            st.sidebar.error(f"Error generating word cloud: {e}")

    if st.sidebar.button("Generate Knowledge Graph"):
        try:
            docs, _ = query_chromadb("", n_results=100)
            if docs:
                generate_knowledge_graph(docs)
            else:
                st.sidebar.info("No documents available to generate knowledge graph.")
        except Exception as e:
            st.sidebar.error(f"Error generating knowledge graph: {e}")

    if prompt := st.chat_input("Ask a question:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.short_term_memory.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.write(prompt)
        
        with st.chat_message("assistant"):
            start_time = time.time()
            response = rag_pipeline(prompt)
            duration = time.time() - start_time
            response_with_duration = f"{response}\n\nDuration: {duration:.2f} seconds"
            st.session_state.messages.append({"role": "assistant", "content": response_with_duration})
            st.session_state.short_term_memory.append({"role": "assistant", "content": response_with_duration})
            st.write(response_with_duration)

if __name__ == "__main__":
    main()