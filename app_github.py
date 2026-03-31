import tempfile
import streamlit as st

from operator import itemgetter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

st.set_page_config(page_title="Document Q&A RAG Chatbot", page_icon="📄", layout="wide")

st.markdown("""
    <style>
        .main-title {
            text-align: center;
            font-size: 36px;
            font-weight: 700;
            color: #1f2937;
            margin-bottom: 0.2rem;
        }
        .sub-title {
            text-align: center;
            font-size: 16px;
            color: #6b7280;
            margin-bottom: 2rem;
        }
        .info-box {
            padding: 12px;
            border-radius: 10px;
            background-color: #f3f4f6;
            color: #111827;
            margin-bottom: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📄 Document Q&A RAG Chatbot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Upload a TXT or PDF file, ask questions, and get grounded answers with conversation history.</div>',
    unsafe_allow_html=True
)

MAX_TOKENS = 100

def count_tokens(messages):
    return sum(len(msg.content.split()) for msg in messages)

def trim_chat_history_by_tokens(messages, max_tokens):
    msgs = messages[:]
    while count_tokens(msgs) > max_tokens and len(msgs) > 1:
        msgs.pop(0)
    return msgs

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

def load_uploaded_document(uploaded_file):
    file_extension = uploaded_file.name.split(".")[-1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_file_path = tmp_file.name

    if file_extension == "txt":
        loader = TextLoader(temp_file_path, encoding="utf-8")
    elif file_extension == "pdf":
        loader = PyPDFLoader(temp_file_path)
    else:
        raise ValueError("Unsupported file type. Please upload a TXT or PDF file.")

    return loader.load()

def build_rag_from_uploaded_file(uploaded_file):
    documents = load_uploaded_document(uploaded_file)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(documents)

    embeddings = get_embeddings()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a helpful Question-Answering assistant.

Use only the retrieved context provided below to answer the user's question.
Do not use outside knowledge or make up information.
If the answer is not present in the retrieved context, reply exactly with:
"I don't know."

Retrieved Context:
{context}"""
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        api_key=st.secrets["GROQ_API_KEY"]
    )

    rag_chain = (
        {
            "context": itemgetter("input") | retriever | RunnableLambda(format_docs),
            "input": itemgetter("input"),
            "chat_history": itemgetter("chat_history")
        }
        | prompt
        | llm
    )

    return rag_chain, len(documents), len(chunks)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "file_name" not in st.session_state:
    st.session_state.file_name = None

if "doc_count" not in st.session_state:
    st.session_state.doc_count = 0

if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

with st.sidebar:
    st.header("⚙️ Controls")
    uploaded_file = st.file_uploader("Upload a TXT or PDF file", type=["txt", "pdf"])

    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.success("Chat history cleared.")

    st.markdown("---")
    st.markdown("### ℹ️ App Info")
    st.caption("Supported files: TXT, PDF")
    st.caption("Embeddings: HuggingFace")
    st.caption("LLM: Groq")
    st.caption("Vector Store: Chroma")

    if st.session_state.file_name:
        st.markdown("---")
        st.markdown("### 📄 Uploaded File")
        st.write(f"**Name:** {st.session_state.file_name}")
        st.write(f"**Documents:** {st.session_state.doc_count}")
        st.write(f"**Chunks:** {st.session_state.chunk_count}")

if uploaded_file is not None:
    if st.session_state.file_name != uploaded_file.name:
        try:
            with st.spinner("Processing uploaded file..."):
                rag_chain, doc_count, chunk_count = build_rag_from_uploaded_file(uploaded_file)
                st.session_state.rag_chain = rag_chain
                st.session_state.file_name = uploaded_file.name
                st.session_state.doc_count = doc_count
                st.session_state.chunk_count = chunk_count
                st.session_state.chat_history = []

            st.success(f"File uploaded and processed successfully: {uploaded_file.name}")

        except Exception as e:
            st.error(f"Error processing file: {e}")
            st.stop()

if st.session_state.rag_chain is None:
    st.markdown("""
    <div class="info-box">
        <b>Welcome!</b><br>
        Upload a <code>.txt</code> or <code>.pdf</code> file from the sidebar to start chatting with your document.<br><br>
        You can:
        <ul>
            <li>Ask factual questions from the document</li>
            <li>Ask follow-up questions</li>
            <li>Maintain conversation history</li>
            <li>Get grounded answers based on retrieved context</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"### Chatting with: `{st.session_state.file_name}`")

    for msg in st.session_state.chat_history:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.markdown(msg.content)
        elif isinstance(msg, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(msg.content)

    user_input = st.chat_input("Ask a question about your uploaded document...")

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)

        try:
            with st.spinner("Generating answer..."):
                response = st.session_state.rag_chain.invoke({
                    "input": user_input,
                    "chat_history": st.session_state.chat_history
                })

            with st.chat_message("assistant"):
                st.markdown(response.content)

            st.session_state.chat_history.append(HumanMessage(content=user_input))
            st.session_state.chat_history.append(AIMessage(content=response.content))

            st.session_state.chat_history = trim_chat_history_by_tokens(
                st.session_state.chat_history,
                MAX_TOKENS
            )

        except Exception as e:
            st.error(f"Error generating response: {e}")
