import os
import tempfile
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from PyPDF2 import PdfReader
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.chains import LLMChain
from langchain.evaluation.criteria import CriteriaEvalChain, Criteria
import pandas as pd

# Load environment variables
load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

if not OPENAI_API_KEY:
    st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    st.stop()

# Streamlit title
st.title("Chat with your PDFs")

# Upload PDFs
uploaded_files = st.file_uploader("Choose PDFs", type="pdf", accept_multiple_files=True)

if not uploaded_files:
    st.warning("Please upload at least one PDF file.")
    st.stop()

# Define a class for documents
class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

# PDF processing function
def load_and_process_pdfs(files):
    documents = []
    file_name_mapping = {}
    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file.read())
            tmp_file_path = tmp_file.name

        # Use PyPDF2 to get the number of pages and their text
        reader = PdfReader(tmp_file_path)
        num_pages = len(reader.pages)

        # Extract text and metadata
        for page_num in range(num_pages):
            page = reader.pages[page_num]
            text = page.extract_text()
            if text:
                document = Document(
                    page_content=text,
                    metadata={
                        "source": tmp_file_path,
                        "file_name": file.name,
                        "page_number": page_num + 1  # Pages are 1-indexed
                    }
                )
                documents.append(document)

        file_name_mapping[tmp_file_path] = file.name

    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    return splits, file_name_mapping

# Initialize the vector store with documents
def initialize_vectorstore(splits):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    return FAISS.from_documents(documents=splits, embedding=embeddings)

# Load and process the PDFs
splits, file_name_mapping = load_and_process_pdfs(uploaded_files)
vectorstore = initialize_vectorstore(splits)

llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")

# Define the prompt template
system_message = SystemMessagePromptTemplate.from_template("You are a helpful assistant that provides detailed and comprehensive answers based on the provided documents.")
human_message = HumanMessagePromptTemplate.from_template("Context from the provided documents:\n{context}\n\nQuestion: {question}\n\nAnswer:")
prompt_template = ChatPromptTemplate.from_messages([system_message, human_message])

# Create the LLMChain
llm_chain = LLMChain(prompt=prompt_template, llm=llm)

# Create the CriteriaEvalChain for evaluation
criteria = {
    Criteria.CONCISENESS: "Is the submission concise and to the point?",
    Criteria.RELEVANCE: "Is the submission referring to a real quote from the text?",
    Criteria.CORRECTNESS: "Is the submission correct, accurate, and factual?",
    Criteria.COHERENCE: "Is the submission coherent, well-structured, and organized?"
}
eval_chain = CriteriaEvalChain.from_llm(llm=llm, criteria=criteria)

# Function to generate response and evaluation
def generate_response(question, context_memory, file_name_mapping):
    retriever = vectorstore.as_retriever()
    docs = retriever.get_relevant_documents(question)

    # Update context memory
    context = "\n\n".join([doc.page_content for doc in docs])
    context_memory += "\n\n" + context

    res = llm_chain.run({"context": context_memory, "question": question})

    answer = res.strip()

    # Extract source information
    sources_metadata = {}
    context_used = []
    for doc in docs:
        try:
            source_info = file_name_mapping.get(doc.metadata['source'], "Unknown source")
            file_name = doc.metadata.get('file_name', 'Unknown file')
            page_number = doc.metadata.get('page_number', 'unknown')
            context_used.append({
                "content": doc.page_content[:200],
                "file_name": file_name,
                "page_number": page_number
            })  # Add snippet for debugging context
            if file_name not in sources_metadata:
                sources_metadata[file_name] = set()
            sources_metadata[file_name].add(page_number)
        except AttributeError:
            st.write(f"Document {doc} does not have the expected metadata structure.")

    # Format sources for better readability
    formatted_sources = []
    for source, pages in sources_metadata.items():
        pages_str = ', '.join(str(page) for page in pages)
        formatted_sources.append(f"{source} (pages: {pages_str})")

    sources_concatenated = '\n'.join(formatted_sources)

    # Evaluate the generated answer
    graded_outputs = eval_chain.evaluate_strings(
        prediction=answer,
        input=question,
    )

    # Ensure graded outputs are in the correct structure
    evaluation_data = []
    for criterion, output in graded_outputs.items():
        if isinstance(output, dict):
            evaluation_data.append({
                "Criterion": criterion.capitalize(),
                "Score": output.get('score', 'N/A'),
                "Reasoning": output.get('reasoning', 'N/A')
            })
        else:
            evaluation_data.append({
                "Criterion": criterion.capitalize(),
                "Score": 'N/A',
                "Reasoning": str(output)
            })

    eval_results_str = "\n\n".join([f"{data['Criterion']}: {data['Reasoning']}" for data in evaluation_data])

    return answer, sources_concatenated, eval_results_str, context_memory

# Initialize context memory
if 'context_memory' not in st.session_state:
    st.session_state.context_memory = ""

# Streamlit interaction
st.session_state.setdefault('messages', [{"role": "assistant", "content": "Ask me a question!"}])

# Handle the chat input and response logic
if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    answer, sources, eval_results, context_memory = generate_response(prompt, st.session_state.context_memory, file_name_mapping)
    st.session_state.context_memory = context_memory  # Update context memory
    st.session_state.messages.append({"role": "assistant", "content": f"{answer}\n\nSources:\n{sources}\n\nEvaluation Results:\n{eval_results}"})

    # Display messages
    for message in st.session_state.messages:
        with st.container():
            role = message["role"]
            if role == "assistant":
                st.info(message["content"])
            else:
                st.success(message["content"])
