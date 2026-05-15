# ☁️ Cloud RAG System — LangChain + Typesense + Groq LLM

A fully end-to-end **Retrieval Augmented Generation (RAG)** pipeline powered by **Typesense Cloud** as the vector database and **Groq's LLaMA 3.1** as the LLM. Load any PDF or text document, embed it into the cloud, and get intelligent answers to your questions — instantly.

---

## 💡 What makes this different from local RAG?

| Feature | Local RAG (ChromaDB) | Cloud RAG (Typesense) |
|--------|----------------------|----------------------|
| Storage | Your machine | Typesense Cloud |
| Scalability | Limited | Production-ready |
| Persistence | Local files | Cloud hosted |
| Access | Single machine | Anywhere |
| Speed | Moderate | Fast (cloud-optimized) |

---

## 🔁 How It Works

```
Your Documents (PDF / TXT)
         ↓
Load with LangChain Loaders
         ↓
Split into Chunks (CharacterTextSplitter)
         ↓
Generate Embeddings (HuggingFace: all-MiniLM-L6-v2)
         ↓
Store in Typesense Cloud Vector Database
         ↓
User asks a Question
         ↓
Similarity Search → Retrieve Relevant Chunks
         ↓
Groq LLaMA 3.1 generates a grounded Answer
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| LangChain | Document loading, splitting, retrieval chain |
| Typesense | Cloud vector database for storing embeddings |
| HuggingFace Embeddings | Free local embedding model |
| Groq + LLaMA 3.1 8B | Fast, free LLM for answer generation |
| PyPDFLoader | PDF parsing |
| python-dotenv | Secure API key management |

---

## 📁 Project Structure

```
cloud-rag-system/
│
├── data/
│   ├── pdf/                   # Place your PDF files here
│   └── text_files/            # Place your text files here
│
├── notebook/
│   └── typesense.ipynb        # Full RAG pipeline notebook
│
├── main.py                    # Entry point
├── requirements.txt           # Dependencies
├── .gitignore                 # Keeps API keys safe
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/Krish00i7/cloud-rag-system.git
cd cloud-rag-system
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up your environment variables
Create a `.env` file in the root folder:
```
GROQ_API_KEY=your_groq_api_key_here
typesense_host=xxx.a1.typesense.net
typesense_api_key=your_typesense_api_key_here
```

- Get Groq API key free at [console.groq.com](https://console.groq.com)
- Get Typesense Cloud free at [cloud.typesense.org](https://cloud.typesense.org)

### 4. Add your documents
- PDFs → `data/pdf/`
- Text files → `data/text_files/`

### 5. Run the notebook
Open `notebook/typesense.ipynb` in Jupyter and run all cells.

---

## ⚙️ Core Implementation

### 📄 Load & Split Documents
```python
loader = DirectoryLoader("../data/pdf", glob="**/*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)
```

### ☁️ Store in Typesense Cloud
```python
embeddings = HuggingFaceEmbeddings()

docsearch = Typesense.from_documents(
    docs,
    embeddings,
    typesense_client_params={
        "host": os.getenv("typesense_host"),
        "port": "443",
        "protocol": "https",
        "typesense_api_key": os.getenv("typesense_api_key"),
        "typesense_collection_name": "lang-chain"
    }
)
```

### 🤖 RAG with Groq LLaMA 3.1
```python
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant",
    temperature=0.1,
    max_tokens=1024
)

def rag_simple(query, retriever, llm, top_k=3):
    results = retriever.invoke(query)[:top_k]
    context = "\n\n".join([doc.page_content for doc in results])
    prompt = f"Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"
    response = llm.invoke(prompt)
    return response.content
```

### 💬 Ask Questions
```python
print(rag_simple("What are the types of machine learning?", retriever, llm))
```

---

## 📌 Key Concepts Learned

- Difference between local and cloud vector databases
- How Typesense stores and retrieves document embeddings via HTTPS
- How HuggingFace embeddings work locally without any API cost
- How to connect a cloud retriever to a Groq LLM for end-to-end RAG
- How context is built from retrieved chunks and passed to an LLM

---

## 🔮 What's Next

- [ ] Add Streamlit chat UI
- [ ] Support multiple document collections
- [ ] Add conversation memory for multi-turn Q&A
- [ ] Evaluate answer quality with retrieval scoring

---

## 👨‍💻 Author

**Krishnakumar M**  
B.Sc Computer Science | SRM Arts and Science College, Chennai  
[GitHub](https://github.com/Krish00i7) • [Email](mailto:krishna00i777@gmail.com)

---

> Unlike local RAG — this pipeline stores embeddings in the cloud, making it scalable, persistent, and production-ready.
