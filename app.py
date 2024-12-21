import os
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.agents import Tool, initialize_agent, AgentType

# Configure paths - remove the extra 's' at the start
MODEL_PATH = "saved_model"
TOKENIZER_PATH = "saved_tokenizer"
VECTORSTORE_PATH = "vectorstore"

# Ensure directories exist
os.makedirs(VECTORSTORE_PATH, exist_ok=True)

# Initialize Streamlit
st.title("Mathematical Chatbot")
st.write("Welcome to the Mathematical Chatbot! Ask me any question.")

try:
    # Load the saved model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    hf_model = AutoModel.from_pretrained(MODEL_PATH)
    st.success("Model and tokenizer loaded successfully!")
except Exception as e:
    st.error(f"Error loading model/tokenizer: {str(e)}")
    st.stop()

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize Chroma vectorstore
try:
    vectorstore = Chroma(
        persist_directory=VECTORSTORE_PATH,
        embedding_function=embeddings
    )
    st.success("Vector store loaded successfully!")
except Exception as e:
    st.error(f"Error loading vector store: {str(e)}")
    st.stop()

# Initialize retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

# Initialize Groq LLM
try:
    llm = ChatGroq(
        temperature=0.7,
        model_name="llama3-8b-8192",
        groq_api_key="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"
    )
except Exception as e:
    st.error(f"Error initializing Groq LLM: {str(e)}")
    st.stop()

# Math tool function with improved error handling
def math_tool_function(query: str) -> str:
    try:
        # Add basic input validation
        if not isinstance(query, str):
            return "Error: Input must be a string"
        if not query.strip():
            return "Error: Empty input"
            
        # Evaluate the expression
        result = eval(query)
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"

# Define math tool
math_tool = Tool(
    name="MathTool",
    func=math_tool_function,
    description="Useful for evaluating mathematical expressions. Input should be a valid Python mathematical expression as a string."
)

# Initialize QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
)

# Initialize agent with better error handling
try:
    agent = initialize_agent(
        tools=[math_tool],
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )
    st.success("Agent initialized successfully!")
except Exception as e:
    st.error(f"Error initializing agent: {str(e)}")
    st.stop()

# Chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input field for user queries
user_input = st.text_input("Ask me a question:")

if user_input:
    with st.spinner("Thinking..."):
        try:
            # Get response from agent
            response = agent.invoke({
                "input": user_input,
                "chat_history": st.session_state.chat_history
            })
            
            # Update chat history
            st.session_state.chat_history.append((user_input, response["output"]))
            
            # Display response
            st.write("### Answer:")
            st.write(response["output"])
            
            # If it's a math question, show the calculation
            if "MathTool" in str(response):
                st.write("### Calculation Details:")
                st.code(response["intermediate_steps"])
                
        except Exception as e:
            st.error(f"Error processing your request: {str(e)}")
            st.write("Please try rephrasing your question.")

# Display chat history
if st.session_state.chat_history:
    st.write("### Chat History:")
    for i, (q, a) in enumerate(st.session_state.chat_history):
        st.write(f"Q{i+1}: {q}")
        st.write(f"A{i+1}: {a}")
        st.write("---")