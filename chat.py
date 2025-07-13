import datetime
import os

import streamlit as st
from dotenv import load_dotenv
from langchain.memory import (ConversationBufferMemory,
                              StreamlitChatMessageHistory)

from chains.conversational_chain import ConversationalChain
from chains.conversational_retrieval_chain import (
    TEMPLATE, ConversationalRetrievalChain)

from streaming import StreamHandler
from free_models import get_groq_llm, get_free_embeddings, get_available_groq_models


@st.cache_resource(show_spinner=True)
def load_knowledge_from_files(uploaded_files):
    from langchain_community.document_loaders import PyPDFLoader, TextLoader
    from langchain_text_splitters import CharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    import tempfile
    
    documents = []
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    for uploaded_file in uploaded_files:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Load document based on file type
        if uploaded_file.name.lower().endswith('.pdf'):
            loader = PyPDFLoader(tmp_file_path)
        elif uploaded_file.name.lower().endswith('.txt'):
            loader = TextLoader(tmp_file_path, encoding='utf-8')
        else:
            continue
            
        docs = loader.load()
        documents.extend(text_splitter.split_documents(docs))
        
        # Clean up temp file
        os.unlink(tmp_file_path)
    
    if documents:
        vectorstore = FAISS.from_documents(documents, get_free_embeddings())
        return vectorstore
    return None


@st.cache_data
def groq_model_list():
    return get_available_groq_models()


class StreamlitChatView:
    def __init__(self) -> None:
        st.set_page_config(page_title="DocBotX", page_icon="🤖", layout="wide")
        with st.sidebar:
            st.title("🤖 DocBotX")
            with st.expander("Model parameters"):
                self.model_name = st.selectbox("Groq Model:", options=groq_model_list())
                self.temperature = st.slider("Temperature", min_value=0., max_value=2., value=0.7, step=0.01)
                self.streaming = st.checkbox("Enable Streaming (slower but real-time)", value=False)
                self.top_p = st.slider("Top p", min_value=0., max_value=1., value=1., step=0.01)
                self.frequency_penalty = st.slider("Frequency penalty", min_value=0., max_value=2., value=0., step=0.01)
                self.presence_penalty = st.slider("Presence penalty", min_value=0., max_value=2., value=0., step=0.01)
            with st.expander("Prompts"):
                curdate = datetime.datetime.now().strftime("%Y-%m-%d")
                model_name = self.model_name.replace('-turbo', '').upper()
                system_message = (f"You are DocBotX, an AI assistant powered by {model_name} via Groq API. "
                                  f"You help users by answering questions about uploaded documents.\n"
                                  f"Current date: {curdate}\n")
                self.system_message = st.text_area("System message", value=system_message)
                self.context_prompt = st.text_area("Context prompt", value=TEMPLATE)

            with st.expander("📁 Upload Documents"):
                self.uploaded_files = st.file_uploader(
                    "Upload PDF or TXT files",
                    type=['pdf', 'txt'],
                    accept_multiple_files=True,
                    help="Upload documents to chat with them"
                )
                self.inject_knowledge = len(self.uploaded_files) > 0 if self.uploaded_files else False
                if self.uploaded_files:
                    st.success(f"📄 {len(self.uploaded_files)} file(s) uploaded")
                    for file in self.uploaded_files:
                        st.write(f"• {file.name}")
        self.user_query = st.chat_input(placeholder="Ask me anything!")

    def add_message(self, message: str, author: str):
        assert author in ["user", "assistant"]
        with st.chat_message(author):
            st.markdown(message)

    def add_message_stream(self, author: str):
        assert author in ["user", "assistant"]
        return StreamHandler(st.chat_message(author).empty())


def setup_memory():
    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    return ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)


def setup_chain(llm, memory, inject_knowledge, system_message, context_prompt, retriever):
    if not inject_knowledge:
        # Custom conversational chain
        return ConversationalChain(
            llm=llm,
            memory=memory,
            system_message=system_message,
            verbose=True)
    else:
        return ConversationalRetrievalChain(
            llm=llm,
            retriever=retriever,
            memory=memory,
            system_message=system_message,
            context_prompt=context_prompt,
            verbose=True)


STREAM = False

# Setup
load_dotenv()
view = StreamlitChatView()
memory = setup_memory()
retriever = None
if view.inject_knowledge and view.uploaded_files:
    try:
        vectorstore = load_knowledge_from_files(view.uploaded_files)
        if vectorstore:
            retriever = vectorstore.as_retriever()
            st.success("✅ Knowledge base loaded successfully!")
        else:
            st.warning("No documents could be processed")
            view.inject_knowledge = False
    except Exception as e:
        st.error(f"Failed to load knowledge base: {str(e)}")
        st.info("Continuing without knowledge injection...")
        view.inject_knowledge = False

# Update Groq LLM with streaming preference
from langchain_groq import ChatGroq
# Get API key from Streamlit secrets or environment
api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

try:
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name=view.model_name,
        temperature=view.temperature,
        streaming=view.streaming
    )
except Exception as e:
    st.error(f"Failed to initialize model {view.model_name}: {str(e)}")
    st.info("Falling back to default model...")
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.1-8b-instant",
        temperature=view.temperature,
        streaming=view.streaming
    )
chain = setup_chain(llm=llm, memory=memory, inject_knowledge=view.inject_knowledge,
                    retriever=retriever, system_message=view.system_message,
                    context_prompt=view.context_prompt)

# Display previous messages
for message in memory.chat_memory.messages:
    view.add_message(message.content, 'assistant' if message.type == 'ai' else 'user')

# Send message
if view.user_query:
    view.add_message(view.user_query, "user")
    if view.streaming:
        st_callback = view.add_message_stream("assistant")
        chain.run({"question": view.user_query}, callbacks=[st_callback])
    else:
        with st.spinner("🤔 Thinking..."):
            try:
                response = chain.run({"question": view.user_query})
                view.add_message(response, "assistant")
            except Exception as e:
                error_msg = str(e)
                if "decommissioned" in error_msg or "model" in error_msg.lower():
                    st.error(f"⚠️ Model Error: {error_msg}")
                    st.info("Please select a different model from the sidebar.")
                else:
                    st.error(f"Error: {error_msg}")
                    view.add_message(f"Sorry, I encountered an error: {error_msg}", "assistant")
