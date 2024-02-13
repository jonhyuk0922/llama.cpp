import streamlit as st
import tiktoken
from loguru import logger
from langchain.chains import ConversationalRetrievalChain

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import LlamaCpp

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import StreamlitChatMessageHistory


def main():
    # emoji : https://github.com/streamlit/streamlit/blob/c023339b80fde00a4578bd32f02e054c9d9d9cf1/lib/streamlit/commands/page_config.py#L50
    st.set_page_config(page_title="RAG system", page_icon="🌀")

    st.title("_RAG system for :blue[Paper.pdf]_ 📖")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    with st.sidebar:
        uploaded_files = st.file_uploader(
            "Upload your file", type=["pdf", "docx", "pptx"], accept_multiple_files=True
        )
        process = st.button("Process")

    if process:
        files_text = get_text(uploaded_files)
        text_chunks = get_text_chunks(files_text)
        vectorstore = get_vectorstore(text_chunks)
        st.session_state.conversation = get_conversation_chain(vectorstore)

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": "Hello, please enter the file and ask me anything.✨",
            }
        ]
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    history = StreamlitChatMessageHistory(key="chat_messages")
    if query := st.chat_input("Please enter a question."):
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            chain = st.session_state.conversation
            with st.spinner("Thinking ..."):
                result = chain({"question": query})
                response = result["answer"]
                source_documents = result["source_documents"]
                st.markdown(response)
                with st.expander("Reference document"):
                    st.markdown(
                        source_documents[0].metadata["source"],
                        help=source_documents[0].page_content,
                    )
                    st.markdown(
                        source_documents[1].metadata["source"],
                        help=source_documents[1].page_content,
                    )
                    st.markdown(
                        source_documents[2].metadata["source"],
                        help=source_documents[2].page_content,
                    )

        st.session_state.messages.append({"role": "assistant", "content": response})


def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)


def get_text(docs):
    doc_list = []

    for doc in docs:
        file_name = doc.name  # doc 객체 이름을 파일 이름으로 그냥 사용
        with open(file_name, "wb") as file:  # 파일을 doc.name으로 저장
            file.write(doc.getvalue())
            logger.info(f"Uploaded {file_name}")

        if ".pdf" in doc.name:
            loader = PyPDFLoader(file_name)
            documents = loader.load_and_split()

        elif ".docx" in doc.name:
            loader = Docx2txtLoader(file_name)
            documents = loader.load_and_split()

        elif ".pptx" in doc.name:
            loader = UnstructuredPowerPointLoader(file_name)
            documents = loader.load_and_split()

        doc_list.extend(documents)

    return doc_list


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,  # 맥락 고려를 위해 겹치는 부분 설정
        length_function=tiktoken_len,
    )
    chunks = text_splitter.split_documents(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    return vectordb


def get_conversation_chain(vetorestore):
    llm = LlamaCpp(
        model_path = "./models/llama-2-7b-chat.Q4_K_M.gguf",
        temperature=0.75,
        n_ctx=4096,
        top_p=1,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        verbose=True,
    )
    
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vetorestore.as_retriever(
            search_type="mmr", vervose=True
        ),  # Maximum marginal relevance
        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True, output_key="answer"
        ),  # only remember answer

        get_chat_history=lambda h: h,
        return_source_documents=True,
        verbose=True,
    )

    return conversation_chain


if __name__ == "__main__":
    main()
