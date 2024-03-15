import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv


load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_conversational_chain():

    prompt_template = """
      
    'INPUT TEXT':
        {context}
        **Question:**
        {question}
    PROMPT: Your role is a ayurvedic doctor bot SwastVeda now what you have to do is analyze the given text given that is delimited by text 'INPUT TEXT' and analyze the Question delimited by 'Question.
    Where format of INPUT TEXT is as: 
      Brief Introduction of the disease, 
      case definition, 
      types of disease with their characterstics, 
      differential diagnosis, 
      3 Levels of that disease: Each level consist of 
        Clinical Diagnosis, 
        Examination, 
        Investigation, 
        Line of treatment 
        Medicines for each level and also medicines according to each types of that disease with proper dosage.
    Now according to the disease that you identified from question provide the user correct ayurvedic medicines,home remedies and yoga poses with proper dosage and timining to consume it. Also provide the do's, dont's and preventions that user must take to recover from the disease.
        """
   
    generation_config = {
        "temperature": 0.7,
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": 2048,
    }
    model = ChatGoogleGenerativeAI(model="gemini-pro", generation_config=generation_config)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    

    print(response)
    st.write("Reply: ", response["output_text"])


def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

if __name__ == "__main__":
    main()
