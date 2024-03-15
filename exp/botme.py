import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from google.generativeai.types import HarmCategory, HarmBlockThreshold,HarmProbability

def get_conversational_chain():
    prompt_template = """
    'INPUT TEXT':
        {context}
        **Question:**
        {question}
    PROMPT:
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
   
    """
   
    generation_config = {
  "temperature": 0.9,
  "top_p": 1,
  "top_k": 1,
  "max_output_tokens": 2048,
}

    safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_ONLY_HIGH"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_ONLY_HIGH"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_ONLY_HIGH"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_ONLY_HIGH"
  },
]
    model = genai.GenerativeModel(model_name="gemini-1.0-pro",
                              generation_config=generation_config,
                              safety_settings=safety_settings)
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
    st.write("Reply: ", response["output_text"])

def main():
    st.set_page_config("Ayurvedic Doctor Chat")
    st.header("Ask Health Questions to Ayurvedic Doctor SwastVeda 💁")
    
    user_questions = []
    for i in range(5):
        question = st.text_input(f"Question {i+1}:")
        if question:
            user_questions.append(question)
    
    if user_questions:
        for question in user_questions:
            user_input(question)

if __name__ == "__main__":
    main()
