# 7th
from flask import Flask, request, jsonify
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = Flask(__name__)

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
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

@app.route('/api/chat', methods=['POST'])
def chat():
    user_question = request.json.get('question')
    if not user_question:
        return jsonify({"error": "Question parameter is missing"}), 400
    response_text = user_input(user_question)
    return jsonify({"response": response_text})

if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0",port=5000)

# 6th
# from flask import Flask, request, jsonify
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv

# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv
# from flask import Flask, request, jsonify
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import os
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# import google.generativeai as genai
# from langchain_community.vectorstores import FAISS
# from langchain_google_genai import ChatGoogleGenerativeAI

# load_dotenv()

# app = Flask(__name__)

# def get_conversational_chain():
#     prompt_template = """
#     'INPUT TEXT':
#         {context}
#         **Question:**
#         {question}
#     PROMPT: Your role is an Ayurvedic doctor bot named SwastVeda. Your task is to ask provide guidance and recommendations based on the analyzed input.

# FORMAT OF INPUT TEXT:
# - Brief Introduction of the disease
# - Case definition
# - Types of disease with their characteristics
# - Differential diagnosis
# - Three levels of the disease, each consisting of:
#   - Clinical Diagnosis
#   - Examination
#   - Investigation
#   - Line of treatment
#   - Medicines for each level and also medicines according to each type of that disease with proper dosage
# Provide the information to user as an response- 
# -Correct Ayurvedic medicines
# - Home remedies
# - Yoga poses with proper dosage and timing
# - Dos and don'ts
# - Precautions for recovery from the disease

#     """

#     generation_config = {
#         "temperature": 0.7,
#         "top_p": 1,
#         "top_k": 1,
#         "max_output_tokens": 2048,
#     }
#     model = ChatGoogleGenerativeAI(model="gemini-pro", generation_config=generation_config)

#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#     chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

#     return chain

# def user_input(user_questions):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
#     # Concatenate user questions into a single string
#     user_input_text = " ".join(user_questions)
    
#     docs = new_db.similarity_search(user_input_text)
#     chain = get_conversational_chain()
    
#     response = chain({"input_documents": docs, "question": user_input_text}, return_only_outputs=True)
#     return response["output_text"]

# @app.route('/chatbot', methods=['POST'])
# def chatbot():
#     data = request.get_json()
#     user_questions = data.get('questions', [])

#     if not user_questions:
#         return jsonify({'error': 'Questions are missing'}), 400

#     response = user_input(user_questions)

#     return jsonify({'questions': user_questions, 'answer': response})

# if __name__ == "__main__":
#     app.run(debug=True)


# # 5th
# from flask import Flask, request, jsonify
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv
# from flask import Flask, request, jsonify
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import os
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# import google.generativeai as genai
# from langchain_community.vectorstores import FAISS
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# load_dotenv()

# app = Flask(__name__)

# def get_conversational_chain():
#     prompt_template = """
#     'INPUT TEXT':
#         {context}
#         **Question:**
#         {question}
#     PROMPT: Your role is an Ayurvedic doctor bot named SwastVeda. Your task is to ask the user 3-5 questions about their disease and then provide guidance and recommendations based on the analyzed input.

# FORMAT OF INPUT TEXT:
# - Brief Introduction of the disease
# - Case definition
# - Types of disease with their characteristics
# - Differential diagnosis
# - Three levels of the disease, each consisting of:
#   - Clinical Diagnosis
#   - Examination
#   - Investigation
#   - Line of treatment
#   - Medicines for each level and also medicines according to each type of that disease with proper dosage

# Now, please ask the following questions about the disease one by one and store the user's response:
# 1. What are the symptoms you're experiencing?
# 2. How long have you been experiencing these symptoms?
# 3. Have you been diagnosed with any other conditions?
# 4. Are you currently taking any medications or supplements?
# """

#     generation_config = {
#         "temperature": 0.7,
#         "top_p": 1,
#         "top_k": 1,
#         "max_output_tokens": 2048,
#     }
#     model = ChatGoogleGenerativeAI(model="gemini-pro", generation_config=generation_config)

#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#     chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

#     return chain

# def get_user_responses():
#     questions = [
#         "What are the symptoms you're experiencing?",
#         "How long have you been experiencing these symptoms?",
#         "Have you been diagnosed with any other conditions?",
#         "Are you currently taking any medications or supplements?"
#     ]
#     responses = []

#     for question in questions:
#         response = input(question + '\n')
#         responses.append(response)

#     return responses

# def get_cure_recommendation(responses):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
#     user_input_text = " ".join(responses)
#     docs = new_db.similarity_search(user_input_text)
#     chain = get_conversational_chain()
    
#     response = chain({"input_documents": docs, "question": user_input_text}, return_only_outputs=True)
#     return response["output_text"]

# @app.route('/chatbot', methods=['POST'])
# def chatbot():
#     user_responses = get_user_responses()
#     cure_recommendation = get_cure_recommendation(user_responses)
#     return jsonify({'cure_recommendation': cure_recommendation})

# if __name__ == "__main__":
#     app.run(debug=True)

# 4th
# from flask import Flask, request, jsonify
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import os
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# import google.generativeai as genai
# from langchain_community.vectorstores import FAISS
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv
# load_dotenv()

# app = Flask(__name__)

# def get_conversational_chain():
#     prompt_template = """
#     'INPUT TEXT':
#         {context}
#         **Question:**
#         {question}
#     PROMPT: Your role is an Ayurvedic doctor bot named SwastVeda. Your task is to ask the user 3-5 questions about their disease and then provide guidance and recommendations based on the analyzed input.

# FORMAT OF INPUT TEXT:
# - Brief Introduction of the disease
# - Case definition
# - Types of disease with their characteristics
# - Differential diagnosis
# - Three levels of the disease, each consisting of:
#   - Clinical Diagnosis
#   - Examination
#   - Investigation
#   - Line of treatment
#   - Medicines for each level and also medicines according to each type of that disease with proper dosage
# Provide the information to user as an response- 
# #-Correct Ayurvedic medicines
# # - Home remedies
# # - Yoga poses with proper dosage and timing
# # - Dos and don'ts
# # - Precautions for recovery from the disease

# """

#     generation_config = {
#         "temperature": 0.7,
#         "top_p": 1,
#         "top_k": 1,
#         "max_output_tokens": 2048,
#     }
#     model = ChatGoogleGenerativeAI(model="gemini-pro", generation_config=generation_config)

#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#     chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

#     return chain

# def user_input(questions):
#     responses = []
#     for question in questions:
#         user_response = input(question + '\n')  # Send the question to the user and get their response
#         responses.append(user_response)
#     return responses

# def get_cure_recommendation(responses):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
#     # Concatenate user responses into a single string
#     user_input_text = " ".join(responses)
    
#     docs = new_db.similarity_search(user_input_text)
#     chain = get_conversational_chain()
    
#     response = chain({"input_documents": docs, "question": user_input_text}, return_only_outputs=True)
#     return response["output_text"]

# @app.route('/chatbot', methods=['POST'])
# def chatbot():
#     # List of questions to ask the user
#     questions = [
#         "What are the symptoms you're experiencing?",
#         "How long have you been experiencing these symptoms?",
#         "Have you been diagnosed with any other conditions?",
#         "Are you currently taking any medications or supplements?"
#     ]

#     # Ask questions and gather responses
#     user_responses = user_input(questions)

#     # Get cure recommendation based on user responses
#     cure_recommendation = get_cure_recommendation(user_responses)

#     return jsonify({'cure_recommendation': cure_recommendation})

# if __name__ == "__main__":
#     app.run(debug=True)

# # 3rd 

# from flask import Flask, request, jsonify
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv

# load_dotenv()

# app = Flask(__name__)

# def get_conversational_chain():
#     prompt_template = """
#     'INPUT TEXT':
#         {context}
#         **Question:**
#         {question}
#     PROMPT: Your role is an Ayurvedic doctor bot named SwastVeda. Your task is to ask the user 3-5 questions about their disease and then provide guidance and recommendations based on the analyzed input.

# FORMAT OF INPUT TEXT:
# - Brief Introduction of the disease
# - Case definition
# - Types of disease with their characteristics
# - Differential diagnosis
# - Three levels of the disease, each consisting of:
#   - Clinical Diagnosis
#   - Examination
#   - Investigation
#   - Line of treatment
#   - Medicines for each level and also medicines according to each type of that disease with proper dosage

# Now, please ask the following questions about the disease one by one and store the user's response:
# 1. What are the symptoms you're experiencing?
# 2. How long have you been experiencing these symptoms?
# 3. Have you been diagnosed with any other conditions?
# 4. Are you currently taking any medications or supplements?
# 5. (Optional) Do you have any specific concerns or questions regarding your condition?
# """

#     generation_config = {
#         "temperature": 0.7,
#         "top_p": 1,
#         "top_k": 1,
#         "max_output_tokens": 2048,
#     }
#     model = ChatGoogleGenerativeAI(model="gemini-pro", generation_config=generation_config)

#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#     chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

#     return chain

# def user_input(user_questions):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
#     # Concatenate user questions into a single string
#     user_input_text = " ".join(user_questions)
    
#     docs = new_db.similarity_search(user_input_text)
#     chain = get_conversational_chain()
    
#     response = chain({"input_documents": docs, "question": user_input_text}, return_only_outputs=True)
#     return response["output_text"]

# @app.route('/chatbot', methods=['POST'])
# def chatbot():
#     data = request.get_json()
#     user_questions = data.get('questions', [])

#     if len(user_questions) < 3 or len(user_questions) > 5:
#         return jsonify({'error': 'Please provide between 3 and 5 questions'}), 400

#     responses = []
#     for i, question in enumerate(user_questions, start=1):
#         response = user_input(question)
#         responses.append({'question': question, 'answer': response})

#     return jsonify({'responses': responses})

# if __name__ == "__main__":
#     app.run(debug=True)

# from flask import Flask, request, jsonify
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import os
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# import google.generativeai as genai
# from langchain_community.vectorstores import FAISS
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv

# load_dotenv()
# os.getenv("GOOGLE_API_KEY")
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# app = Flask(__name__)

# def get_conversational_chain():
#     prompt_template = """
#     'INPUT TEXT':
#         {context}
#         **Question:**
#         {question}
#     PROMPT: PROMPT: Your role is an Ayurvedic doctor bot named SwastVeda. Your task is to ask the user 3-4 questions about their disease and then provide guidance and recommendations based on the analyzed input.

# FORMAT OF INPUT TEXT:
# - Brief Introduction of the disease
# - Case definition
# - Types of disease with their characteristics
# - Differential diagnosis
# - Three levels of the disease, each consisting of:
#   - Clinical Diagnosis
#   - Examination
#   - Investigation
#   - Line of treatment
#   - Medicines for each level and also medicines according to each type of that disease with proper dosage
# Now, please ask the following questions about the disease one by one store the users response and then provide the information- Correct Ayurvedic medicines
# - Home remedies
# - Yoga poses with proper dosage and timing
# - Dos and don'ts
# - Precautions for recovery from the disease
# :Questions you nedd to ask
# 1. What are the symptoms you're experiencing?
# 2. How long have you been experiencing these symptoms?
# 3. Have you been diagnosed with any other conditions?
# 4. Are you currently taking any medications or supplements?

#     """

#     generation_config = {
#         "temperature": 0.7,
#         "top_p": 1,
#         "top_k": 1,
#         "max_output_tokens": 2048,
#     }
#     model = ChatGoogleGenerativeAI(model="gemini-pro", generation_config=generation_config)

#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#     chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

#     return chain

# def user_input(user_question):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
#     docs = new_db.similarity_search(user_question)
#     chain = get_conversational_chain()
#     response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
#     return response["output_text"]

# @app.route('/chatbot', methods=['POST'])
# def chatbot():
#     data = request.get_json()
#     user_question = data.get('question', '')

#     if not user_question:
#         return jsonify({'error': 'Question is missing'}), 400

#     response = user_input(user_question)

#     return jsonify({'question': user_question, 'answer': response})

# if __name__ == "__main__":
#     app.run(debug=True)


# from flask import Flask, request, jsonify
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv

# load_dotenv()

# app = Flask(__name__)

# def get_conversational_chain():
#     prompt_template = """
# 'INPUT TEXT':
#     {context}
#     **Question:**
#     {question}
# PROMPT: PROMPT: Your role is an Ayurvedic doctor bot named SwastVeda. Your task is to ask the user 3-4 questions about their disease and then provide guidance and recommendations based on the analyzed input.

# FORMAT OF INPUT TEXT:
# - Brief Introduction of the disease
# - Case definition
# - Types of disease with their characteristics
# - Differential diagnosis
# - Three levels of the disease, each consisting of:
#   - Clinical Diagnosis
#   - Examination
#   - Investigation
#   - Line of treatment
#   - Medicines for each level and also medicines according to each type of that disease with proper dosage

# Now, please ask the following questions about the disease one by one and store the user's responses:
# 1. What are the symptoms you're experiencing?
# 2. How long have you been experiencing these symptoms?
# 3. Have you been diagnosed with any other conditions?
# 4. Are you currently taking any medications or supplements?

# Once you have gathered the necessary information, provide the following guidance and recommendations:
# - Correct Ayurvedic medicines
# - Home remedies
# - Yoga poses with proper dosage and timing
# - Dos and don'ts
# - Precautions for recovery from the disease

# """

#     generation_config = {
#         "temperature": 0.7,
#         "top_p": 1,
#         "top_k": 1,
#         "max_output_tokens": 2048,
#     }
#     model = ChatGoogleGenerativeAI(model="gemini-pro", generation_config=generation_config)

#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#     chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

#     return chain

# def user_input(user_questions):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
#     # Concatenate user questions into a single string
#     user_input_text = " ".join(user_questions)
    
#     docs = new_db.similarity_search(user_input_text)
#     chain = get_conversational_chain()
    
#     response = chain({"input_documents": docs, "question": user_input_text}, return_only_outputs=True)
#     return response["output_text"]

# @app.route('/chatbot', methods=['POST'])
# def chatbot():
#     data = request.get_json()
#     user_questions = data.get('questions', [])

#     if not user_questions:
#         return jsonify({'error': 'Questions are missing'}), 400

#     response = user_input(user_questions)

#     return jsonify({'questions': user_questions, 'answer': response})

# if __name__ == "__main__":
#     app.run(debug=True)
