# cs5914-project
A Chatbox base on Chatgpt

For the Frontend:
we use nginx(html,css)

For the Backend
we use flask(openai,blockchain,db)

# Prerequisites

Before you get started, make sure you have the following prerequisites installed on your system:

Python 3.x
pip (Python package manager)
Installation

Clone the SafetyChat repository to your local machine using Git: <br />
bash git clone https://github.com/kennychen126/cs5914-project.git <br />
Change your current working directory to the SafetyChat project folder: <br />
bash cd cs5914-project <br />
Install the required Python dependencies by running the following command: <br />
bash pip install -r requirements.txt <br />
This command will install all the necessary libraries and packages for the project.

# Configuration

Open the llm.py file in your preferred text editor or integrated development environment (IDE).
Replace the openai.api_key with your OpenAI API key. You can obtain an API key by signing up for an account on the OpenAI platform.
Configure other settings as needed for your specific use case.

# Usage

To run the SafetyChat application, follow these steps:

Make sure you have completed the installation and configuration steps as described above.
Open a terminal or command prompt and navigate to the project directory.
Start the Flask application by running the following command: <br />
bash python llm.py <br />
The application should now be running. You can access the chatbot through a web interface.
Open a web browser and navigate to http://127.0.0.1:5011/ to access the chatbot interface.
You can interact with the chatbot by entering personal information and daily tasks, and then sending messages or questions to receive responses.

## Group Member
Kaiyi Chen kennychen@vt.edu

Andrew Fang fafaaf61@vt.edu

Hossein Naderi hnaderi@vt.edu

## Introduction
SafetyChat is a chatbot project designed to assist safety trainers in improving the behavior safety of construction workers by providing them with accurate and relevant safety information based on the content they provide. This guide will help you implement the SafetyChat project on your own system.

The project uses blockchain and prompts to adjust chatgpt so that it can provide more reliable answers based on professional data knowledge.

Chatgpt：
The reason we use chatgpt is because llama2 requires sufficient gpu resources. Although we successfully used llama2 locally, the server resources were insufficient and we could only use chatgpt.

Prompts：
Adaptation for chatgpt's answer.

Blockchain:
You can create a db through embedding and provide information about the problem.
