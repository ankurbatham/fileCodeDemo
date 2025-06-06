import streamlit as st
import os
import tempfile
import pickle
import hashlib
from typing import List, Dict, Any
import logging
from pathlib import Path
import sys

# Fix for PyTorch/Streamlit compatibility issue
import torch
if hasattr(torch, '_classes') and hasattr(torch._classes, '__path__'):
    # Monkey patch to fix the torch._classes.__path__ issue
    if not hasattr(torch._classes.__path__, '_path'):
        torch._classes.__path__._path = []

# Import the updated chatbot components
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory  # Keep original import
from langchain_groq.chat_models import ChatGroq
from concurrent.futures import ThreadPoolExecutor
import warnings

# Suppress deprecation warnings for cleaner output
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Suppress FAISS GPU warnings
import logging
logging.getLogger("faiss").setLevel(logging.ERROR)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@st.cache_resource
def get_cached_models(_api_key: str):
    """Initialize models with caching - Note: underscore prefix to avoid hashing"""
    try:
        # Initialize embedding model with updated import
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'batch_size': 16}
        )
        
        # Initialize LLM with updated model
        llm = ChatGroq(
            model="llama-3.1-8b-instant",  # Updated to supported model
            api_key=_api_key,
            temperature=0.1,
            max_tokens=1024
        )
        
        return embedding_model, llm
    except Exception as e:
        st.error(f"Error initializing models: {e}")
        return None, None

class StreamlitPDFChatbot:
    def __init__(self, api_key: str, cache_dir: str = "./streamlit_cache"):
        self.api_key = api_key
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize session state for memory if not exists
        if 'conversation_memory' not in st.session_state:
            # Initialize memory with proper configuration
            st.session_state.conversation_memory = ConversationBufferMemory(
                memory_key="chat_history", 
                return_messages=True,
                output_key="answer"  # Specify which output to store in memory
            )
        
        self.memory = st.session_state.conversation_memory
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize the models"""
        self.embedding_model, self.llm = get_cached_models(self.api_key)

    def _get_file_hash(self, uploaded_files) -> str:
        """Generate hash for uploaded files"""
        hash_content = ""
        for file in uploaded_files:
            # Use file name and size for hash
            hash_content += f"{file.name}_{file.size}"
        return hashlib.md5(hash_content.encode()).hexdigest()
    
    def _save_uploaded_files(self, uploaded_files) -> List[str]:
        """Save uploaded files to temporary directory"""
        temp_dir = tempfile.mkdtemp()
        file_paths = []
        
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            file_paths.append(file_path)
        
        return file_paths
    
    def _load_single_pdf(self, pdf_path: str) -> List[Any]:
        """Load a single PDF file"""
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            return docs
        except Exception as e:
            st.error(f"Error loading {os.path.basename(pdf_path)}: {e}")
            return []
    
    def load_pdfs_parallel(self, pdf_files: List[str]) -> List[Any]:
        """Load multiple PDFs in parallel with progress bar"""
        documents = []
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with ThreadPoolExecutor(max_workers=min(4, len(pdf_files))) as executor:
            futures = [executor.submit(self._load_single_pdf, pdf) for pdf in pdf_files]
            
            for i, future in enumerate(futures):
                result = future.result()
                documents.extend(result)
                
                # Update progress
                progress = (i + 1) / len(pdf_files)
                progress_bar.progress(progress)
                status_text.text(f"Loaded {i + 1}/{len(pdf_files)} files...")
        
        progress_bar.empty()
        status_text.empty()
        
        return documents
    
    def process_documents(self, documents: List[Any]) -> List[Any]:
        """Process documents with progress tracking"""
        with st.spinner("Processing documents..."):
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            
            chunks = text_splitter.split_documents(documents)
            st.success(f"Created {len(chunks)} text chunks")
            return chunks
    
    def create_vectorstore(self, chunks: List[Any], file_hash: str):
        """Create or load cached vectorstore"""
        cache_path = os.path.join(self.cache_dir, f"vectorstore_{file_hash}")
        
        if os.path.exists(cache_path):
            with st.spinner("Loading cached vectorstore..."):
                try:
                    vectorstore = FAISS.load_local(
                        cache_path, 
                        self.embedding_model,
                        allow_dangerous_deserialization=True
                    )
                    st.success("Loaded from cache!")
                except Exception as e:
                    st.warning(f"Cache loading failed: {e}. Creating new vectorstore...")
                    vectorstore = self._create_new_vectorstore(chunks, cache_path)
        else:
            vectorstore = self._create_new_vectorstore(chunks, cache_path)
        
        return vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
    
    def _create_new_vectorstore(self, chunks, cache_path):
        """Create new vectorstore and cache it"""
        with st.spinner("Creating vectorstore (this may take a while for large files)..."):
            vectorstore = FAISS.from_documents(chunks, self.embedding_model)
            
            # Save to cache
            try:
                vectorstore.save_local(cache_path)
                st.success("Vectorstore cached for future use!")
            except Exception as e:
                st.warning(f"Failed to cache vectorstore: {e}")
            
            return vectorstore
    
    def setup_documents(self, uploaded_files) -> bool:
        """Setup documents and create retriever"""
        try:
            if not uploaded_files:
                st.error("No files uploaded")
                return False
            
            # Generate hash for caching
            file_hash = self._get_file_hash(uploaded_files)
            
            # Save uploaded files
            file_paths = self._save_uploaded_files(uploaded_files)
            
            # Load documents
            st.info("Loading PDF files...")
            documents = self.load_pdfs_parallel(file_paths)
            
            if not documents:
                st.error("No documents loaded successfully")
                return False
            
            st.success(f"Loaded {len(documents)} pages from {len(uploaded_files)} files")
            
            # Process documents
            chunks = self.process_documents(documents)
            
            # Create retriever
            retriever = self.create_vectorstore(chunks, file_hash)
            
            # Store in session state
            st.session_state.retriever = retriever
            st.session_state.files_processed = True
            
            # Cleanup temporary files
            for file_path in file_paths:
                try:
                    os.remove(file_path)
                except:
                    pass
            
            return True
            
        except Exception as e:
            st.error(f"Error setting up documents: {e}")
            return False
    
    def get_response(self, question: str) -> str:
        """Get chatbot response"""
        if 'retriever' not in st.session_state:
            return "Error: No documents loaded. Please upload and process documents first."
        
        try:
            # Create QA chain with updated syntax
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=st.session_state.retriever,
                memory=self.memory,
                return_source_documents=True,
                verbose=False
            )
            
            # Get response
            with st.spinner("Thinking..."):
                result = qa_chain.invoke({"question": question})
            
            # Format response
            response = result['answer']
            
            # Add source information
            if 'source_documents' in result and result['source_documents']:
                sources = set()
                for doc in result['source_documents'][:3]:
                    if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                        sources.add(os.path.basename(doc.metadata['source']))
                
                if sources:
                    response += f"\n\n📚 **Sources:** {', '.join(sources)}"
            
            return response
            
        except Exception as e:
            return f"Error processing your question: {str(e)}"

def main():
    st.set_page_config(
        page_title="PDF Chatbot",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 PDF Chatbot")
    st.markdown("Upload your PDF files and chat with them using AI!")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input (uncomment to use interactive input)
        api_key = st.text_input(
            "🔑 Enter your Groq API Key",
            type="password",
            help="Get your API key from https://console.groq.com/"
            # value="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"  # Remove this default value in production
        )
    
        if not api_key:
            st.warning("Please enter your Groq API key to continue")
            st.stop()
        
        st.success("API Key provided!")
        
        # File upload
        st.header("📁 Upload Files")
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type="pdf",
            accept_multiple_files=True,
            help="Upload one or more PDF files to chat with"
        )
        
        if uploaded_files:
            st.success(f"Uploaded {len(uploaded_files)} file(s)")
            for file in uploaded_files:
                st.write(f"• {file.name} ({file.size/1024/1024:.2f} MB)")
        
        # Process files button
        if uploaded_files and st.button("🚀 Process Files", type="primary"):
            chatbot = StreamlitPDFChatbot(api_key)
            if chatbot.setup_documents(uploaded_files):
                st.success("Files processed successfully! You can now start chatting.")
                st.session_state.chatbot = chatbot
            else:
                st.error("Failed to process files. Please try again.")
        
        # Clear conversation button
        if st.button("🗑️ Clear Conversation"):
            if 'conversation_memory' in st.session_state:
                st.session_state.conversation_memory.clear()
            if 'messages' in st.session_state:
                st.session_state.messages = []
            st.success("Conversation cleared!")
            st.rerun()
    
    # Main chat interface
    if 'files_processed' in st.session_state and st.session_state.files_processed:
        st.header("💬 Chat with your PDFs")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your PDFs..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get bot response
            if 'chatbot' in st.session_state:
                response = st.session_state.chatbot.get_response(prompt)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.markdown(response)
            else:
                st.error("Chatbot not initialized. Please process files first.")
    
    else:
        # Instructions when no files are processed
        st.info("👈 Please upload PDF files and click 'Process Files' to start chatting!")
        
        # Show example questions
        st.header("💡 Example Questions")
        st.markdown("""
        Once you upload your PDFs, you can ask questions like:
        - What is the main topic of this document?
        - Can you summarize the key points?
        - What are the conclusions mentioned?
        - Explain the methodology used
        - What are the recommendations?
        """)
        
        # Show features
        st.header("✨ Features")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🚀 Performance**
            - Parallel processing for multiple files
            - Intelligent caching system
            - Optimized for large documents
            - Fast response times
            """)
        
        with col2:
            st.markdown("""
            **🎯 Capabilities**
            - Multiple PDF support
            - Conversation memory
            - Source attribution
            - Context-aware responses
            """)

if __name__ == "__main__":
    main()