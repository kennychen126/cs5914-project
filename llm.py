import openai
import torch
from langchain.vectorstores import Chroma
import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from huggingface_hub import login
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from transformers import AutoTokenizer, TextStreamer, pipeline, AutoModelForCausalLM
from langchain.llms.huggingface_pipeline import HuggingFacePipeline

# add token
openai.api_key = os.getenv("OPENAI_API_Key")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
login(token=os.getenv("Huggingface"))
# setting for the app
app = Flask(__name__)
CORS(app)
# setting for model
embeddings = HuggingFaceInstructEmbeddings(cache_folder="./embeddings",
                                           model_name="sentence-transformers/all-MiniLM-L6-v2",
                                           model_kwargs={"device": DEVICE})
db3 = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# prompt
#########
prompt_template = """You are a safety trainer aiming to improve behavior safety of construction workers. Do the followings:
                      1-Read the document carefully and accurately
                      2-Read personal information of worker and his daily tasks
                      3-Don't make general information. Match the personal information with safety regulations in the document and inform worker about safety protocols, numbers, and regulations.


{context}

Question: {question}
Answer:"""

#######
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')
qa = ConversationalRetrievalChain.from_llm(
    llm=OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0.1, openai_api_key=openai.api_key),
    memory=memory,
    retriever=db3.as_retriever(search_kwargs={"k": 4}),
    combine_docs_chain_kwargs={'prompt': PROMPT}
)

print("backend is ready")


def get_chatgpt4(user_message):
    return qa({"question": user_message})['answer']


def get_chatllama(user_message):
    return qa({"question": user_message})['answer']




@app.route("/chat_gpt4", methods=["POST"])
def chat_gpt4():
    user_message = request.json.get("message")
    if user_message is None:
        return jsonify({"error": "No message provided"}), 400

    response_message = get_chatgpt4(user_message)

    return jsonify({"message": response_message})


@app.route("/chat_llame2", methods=["POST"])
def chat_llame2():
    user_message = request.json.get("message")
    if user_message is None:
        return jsonify({"error": "No message provided"}), 400

    response_message = get_chatllama(user_message)

    return jsonify({"message": response_message})


if __name__ == "__main__":
    app.run(debug=True, port=5011)
