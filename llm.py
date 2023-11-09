import cv2
import openai
from langchain.vectorstores import Chroma
import os
from flask import Flask, request, jsonify, render_template, Response, make_response
from flask_cors import CORS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# add token
openai.api_key = "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"

# setting for the app
app = Flask(__name__)
CORS(app)
# setting for model
embeddings = OpenAIEmbeddings(openai_api_key="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")
db3 = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# prompt
########
prompt_template = """You are a safety trainer aiming to improve behavior safety of construction workers. Do the followings:
                      1-Read the document carefully and accurately
                      2-Read personal information and daily tasks
                      3-Don't make general information. Match the personal information and daily tasks with safety regulations in the document and inform worker about safety protocols, numbers, and regulations.


{context}
Personal Information: {project}
Daily Tasks: {tasks}
Question: {question}
Answer:"""

prompt_template2 ="""
You are a safety trainer for workers. Base on the const_worker,project and task below. Design a multiple choice questions with 4 options, mostly focused on essential stuff workers need to remember when they are working. Please use less than 100 words to explain the correct choice.
Context:{context}
Project:{field2}
task:{field3}
Please create a different question than the previous one
History:{chat_history}
Please answer in the following format
Question:
A: 
B: 
C: 
D: 
Right answer:
Reason:

"""

#######
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question", "project", "tasks"]
)
PROMPT2 = PromptTemplate(
    template=prompt_template2, input_variables=["context", "field1","field2","field3"]
)
###
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, input_key="question",output_key='answer')
memory2 = ConversationBufferMemory(memory_key='chat_history', return_messages=True, input_key="field3",output_key='answer')
qa = ConversationalRetrievalChain.from_llm(
    llm=OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0.1, openai_api_key=openai.api_key),
    memory=memory,
    retriever=db3.as_retriever(search_kwargs={"k": 2}),
    combine_docs_chain_kwargs={'prompt': PROMPT}
)
qa2 = ConversationalRetrievalChain.from_llm(
    llm=OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0.5, openai_api_key=openai.api_key),
    memory = memory2,
    retriever=db3.as_retriever(search_kwargs={"k": 2}),
    combine_docs_chain_kwargs={'prompt': PROMPT2}
)

def get_chatgpt4(question, user_projects, user_tasks):
    temp = qa({"question": question, "project": user_projects, "tasks": user_tasks})['answer']
    return temp

def create_question(user_worker,user_projects,user_tasks):
    temp2 = qa2({"question":"","field1": user_worker,"field2":user_projects,"field3":user_tasks})['answer']
    temp3 = temp2.splitlines()
    return temp3

@app.route('/')
def hello():
    return 'Hello World'

@app.route("/quiz", methods=["POST"])
def question_gpt():
    user_worker = request.json.get("const_worker")
    user_projects = request.json.get("projects")
    user_tasks = request.json.get("tasks")
    for i in range(10):
        try:
            response_message = create_question(user_worker,user_projects,user_tasks)
            temp3 = []
            for i in response_message:
                if i != '':
                    temp3.append(i)
            print(temp3)
            a0 = temp3[0].split(': ', 1 )[1]
            a1 = temp3[1].split(': ', 1 )[1]
            a2 = temp3[2].split(': ', 1 )[1]
            a3 = temp3[3].split(': ', 1 )[1]
            a4 = temp3[4].split(': ', 1 )[1]
            a5 = temp3[5].split(': ', 1 )[1]
            a6 = temp3[6].split(': ', 1 )[1]
        except:
            continue
        break
    
    return jsonify({"question": a0,"A":a1,"B":a2,"C":a3,"D":a4,"Ans":a5,"res":a6})
    
@app.route("/chat_gpt4", methods=["POST"])
def chat_gpt4():
    question =  request.json.get("message")
    user_worker = request.json.get("const_worker")
    user_projects = request.json.get("projects")
    user_tasks = request.json.get("tasks")
    user_injury = request.json.get("injury")
    user_start = request.json.get("start_t")
    user_end = request.json.get("end_t")

    response_message = get_chatgpt4(question, user_projects, user_tasks)
    return jsonify({"message": response_message})

@app.route("/intro", methods=["POST"])
def intro():
    user_worker = request.json.get("const_worker")
    user_projects = request.json.get("projects")
    user_tasks = request.json.get("tasks")
    user_injury = request.json.get("injury")
    user_start = request.json.get("start_t")
    user_end = request.json.get("end_t")
    temp = ""
    if "Yes" in user_worker:
        if "Yes" in user_injury:
            temp = "I'm a construction worker in "+user_projects+" projects. My upcoming task is "+user_tasks+". I have an injury experience in with "+user_injury+". I'm woking on this task "+user_start+" to "+user_end
        else:
            temp = "I'm a construction worker in "+user_projects+" projects. My upcoming task is "+user_tasks+". I'm woking on this task "+user_start+" to "+user_end
        response2 = get_chatgpt4("Can you give me training based off of the info provided?", user_projects, user_tasks)
        return jsonify({"message": temp + response2})

    else:
        return jsonify({"message": "I only assist construction workers, contact other specialists for geting safety advice in other industries"})

if __name__ == "__main__":
    app.run(host = '0.0.0.0', port=5011)
