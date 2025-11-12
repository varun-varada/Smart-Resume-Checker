# streamlit_app.py
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
# from langchain.prompts import PromptTemplate
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

from langchain_core.output_parsers import StrOutputParser
import os, tempfile, streamlit as st
from dotenv import load_dotenv

load_dotenv()

os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['LANGCHAIN_API_KEY'] = os.getenv('Langchain_Api_key')

st.title("Smart ATS ↔ Job Description Match")

# inputs
job_title = st.text_input("Enter the Job Title")
knowledge_base = st.file_uploader("Upload Resume (PDF)", type=".pdf")
suggestions = st.selectbox("Add Suggestions and improvements ?", options=["Yes", "No"])
description = st.text_area("Enter the Job Description")
button = st.button("Get Match Score")

def load_resume_to_text(uploaded_file):
    """Save uploaded file to a temp file and load with PyMuPDFLoader."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name
    loader = PyMuPDFLoader(tmp_path)
    docs = loader.load()
    return docs  # list of Document objects

if button:
    # basic validation
    if not job_title.strip() or knowledge_base is None or not description.strip():
        st.warning("Please provide job Title, resume file and job description.")
        st.stop()

    # load resume PDF
    docs = load_resume_to_text(knowledge_base)

    # split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)

    # embeddings + vectorstore
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vector = FAISS.from_documents(chunks, embedding)

    # retriever
    retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # If user wants suggestions, set text; else empty
    suggestions_text = ("Yes. Short: provide suggestions and improvements to enhance the resume's "
                        "alignment with the job description.") if suggestions == "Yes" else "No suggestions requested."

    # get the top-k documents for the job description
    retrieved_docs = retriever.invoke(description)
    # join retrieved docs into a single resume_content string (you can keep doc separators)
    resume_content = "\n\n---\n\n".join([d.page_content for d in retrieved_docs])

    # prompt template: include all variables we want to pass
    prompt_template = """
You are an expert HR professional and ATS checker. Your task is to evaluate how well a candidate's resume matches a specific job description.
Generate a match score between 1 and 100, state a sentiment (Positive/Neutral/Negative), and provide a brief explanation.

Consider the following criteria:
1. Relevance of Skills
2. Experience
3. Education
4. Achievements / Certifications
5. Overall Fit

Use the following format exactly:
Match Score: <score between 1-100>
Sentiment: <Positive/Neutral/Negative>
Explanation: <brief explanation>

Job Title: {job_title}
Job Description: {description}

Suggestions and Improvements: {suggestions}
Resume Content: {resume_content}
"""

    prompt = PromptTemplate(input_variables=["job_title", "description", "suggestions", "resume_content"],
                            template=prompt_template)

    # LLM init
    llm = ChatGoogleGenerativeAI(google_api_key=os.getenv('GOOGLE_API_KEY'), model="gemini-2.5-flash")

    # Build a simple chain: prompt -> llm -> string parser
    # We use the chain composition operator: prompt | llm | output_parser
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser

    # invoke chain by passing a dict with all required variables
    with st.spinner("Calculating Match Score..."):
        result = chain.invoke({
            "job_title": job_title,
            "description": description,
            "suggestions": suggestions_text,
            "resume_content": resume_content
        })

    st.subheader("Match Score Result")
    st.write(result)




