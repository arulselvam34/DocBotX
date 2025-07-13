# 📄 Document Chat with Groq

A powerful document chatbot built with Streamlit and Groq API that allows you to upload documents and chat with them using advanced language models.

## 🚀 Features

- **Document Upload**: Support for PDF and TXT files
- **Multiple AI Models**: Choose from various Groq models (Llama 3.1, Llama 3)
- **RAG (Retrieval Augmented Generation)**: Chat with your documents using vector embeddings
- **Fast Responses**: Optimized for speed with optional streaming
- **User-friendly Interface**: Clean Streamlit interface

## 🛠️ Setup

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd rag-chatgpt
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up environment variables
Create a `.env` file in the root directory:
```
GROQ_API_KEY=your_groq_api_key_here
```

Get your free Groq API key from: https://console.groq.com/keys

### 4. Run the application
```bash
streamlit run chat.py
```

## 🌐 Live Demo

[Deploy on Streamlit Cloud](https://share.streamlit.io/)

## 📋 Usage

1. **Upload Documents**: Use the sidebar to upload PDF or TXT files
2. **Select Model**: Choose your preferred Groq model
3. **Adjust Settings**: Modify temperature and other parameters
4. **Start Chatting**: Ask questions about your uploaded documents

## 🔧 Configuration

- **Temperature**: Controls response creativity (0.0 - 2.0)
- **Streaming**: Enable for real-time responses (slower)
- **Models Available**:
  - `llama-3.1-8b-instant` (Fastest)
  - `llama3-8b-8192` (Reliable)
  - `llama3-70b-8192` (Most Capable)

## 🤝 Contributing

Feel free to open issues and pull requests!

## 📄 License

MIT License