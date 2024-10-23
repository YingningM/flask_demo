import os
import re
import csv
import uuid
import json
import pickle
import threading
from datetime import datetime
from operator import itemgetter
from flask import Flask, request, jsonify, render_template, session
from flask_session import Session
from langchain import LLMChain
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
#from langchain_google_vertexai import VertexAI
#from langchain_google_vertexai.model_garden import ChatAnthropicVertex
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory
)
from langchain_core.runnables.history import RunnablePassthrough, RunnableWithMessageHistory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Set the GOOGLE_APPLICATION_CREDENTIALS environment variable
os.environ["OPENAI_BASE_URL"] = 'https://api.bianxie.ai/v1'
os.environ["OPENAI_API_KEY"] = 'sk-FEDzYSgWNKkHgc7labPUIfiFPpI3RZ1Wd7lR6fg2yZ1nNefP'
os.makedirs("conversation_history", exist_ok=True)
persist_directory = "./parentDB_mice"
with open("mice_docstore.pkl", "rb") as file:
    store_dict = pickle.load(file)

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Set a secret key for session management
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

chatbot_instances = {}

CSV_FILE = 'chat_data_app1.csv'
csv_lock = threading.Lock()

if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            '用户ID',
            '用户原提问',
            'LLM1优化后的提问',
            '检索获取的母文档',
            'LLM2生成的回答',
            '收到问题的时间点',
            '生成回复的时间点',
            '是否来自剪切板'
        ])


class BasicChatbot:
    
    def __init__(self, session_id):
        self.store = {}
        self.session_id = session_id
        self.chat_model = ChatOpenAI(
            model = 'gpt-3.5-turbo',
            max_tokens=1024,
            temperature = 0.1,
            top_p = 0.1
            )
        self.embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
        self.retriever, self.rewrite_with_history, self.chain_with_history = self.setup_chain()
    
    def setup_chain(self):
        try:
            db = Chroma(
                 collection_name="mice",
                 embedding_function=self.embeddings,
                 persist_directory=persist_directory
                 )
            store = InMemoryStore()
            store.mset(list(store_dict.items()))
            child_splitter = RecursiveCharacterTextSplitter(chunk_size=600)
            retriever = ParentDocumentRetriever(
                vectorstore=db,
                docstore=store,
                child_splitter=child_splitter,
                search_kwargs={"k": 5}
                )
            system_prompt = ("""You are a Q&A agent that assists customers with inquiries. Your main objectives are to provide accurate information, address customer concerns, and enhance the shopping experience.\ 
                             In your response, please strictly follow the following instructions: 
                             1. ALWAYS relate the question to previous interactions in the conversation history to think about what exactly is the query asking.\
                             2. Assess the documents in the context to determine their usefulness in answering the user's question. Not all documents will be relevant.\
                             3. Base your answers on BOTH your prior knowledge and facts from the context. \
                             4. DO NOT specifically mention the context and documents within in your response.\
                             5. If you do not know the answer to a question, say that you don't know.\
                             Context:
                             {context}"""
                             )
            qa_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    MessagesPlaceholder(variable_name="history"),
                    ("human", "{question}"),
                ]
            )
            rewrite_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", """Please return a new question with the following requirements:
1. If there are existential reference (eg.: it, them) or conditions missing in the question, please make a complete question according to the context.
2. If the question is complete, please keep the original question.
3. You're just rewriting. Don't answer the questions.
4. Finally return json
                     
HISTORY:
[]
NOW QUESTION: Hello, how are you?
THOUGHT: The question "Hello, how are you?" is already clear and complete. There are no pronouns or missing conditions that need to be resolved. => NEED COREFERENCE RESOLUTION: No => OUTPUT QUESTION: Hello, how are you?
JSON: {{ "Completed Question": "Hello, how are you?" }}
-------------------
HISTORY:
[Human: Is Milvus a vector database?
AI: Yes, Milvus is a vector database.]
NOW QUESTION: How to use it?
THOUGHT: The pronoun "it" in the question "How to use it?" refers to "Milvus" mentioned in the previous conversation. Replacing "it" with "Milvus" will make the question complete. => NEED COREFERENCE RESOLUTION: Yes => OUTPUT QUESTION: How to use Milvus?
JSON: {{ "Completed Question": "How to use Milvus?" }}
-------------------
"""),
                    MessagesPlaceholder(variable_name="history"),
                    ("human", "NOW QUESTION: {question}\nTHOUGHT:"),
                ]
            )

            rewrite_chain = LLMChain(
                llm=self.chat_model,
                prompt=rewrite_prompt,
            )
            rewrite_with_history = RunnableWithMessageHistory(
                rewrite_chain,
                self.get_session_history,
                input_messages_key="question",
                history_messages_key="history"
            )

            output_parser = StrOutputParser()
            context = itemgetter("question") | retriever 
            first_step = RunnablePassthrough.assign(context=context)
            #rag_chain = rewrite_chain | first_step | qa_prompt | self.chat_model | output_parser
            #rag_chain = first_step | qa_prompt | self.chat_model | output_parser
            rag_chain = qa_prompt | self.chat_model | output_parser
            chain_with_history = RunnableWithMessageHistory(
                rag_chain,
                self.get_session_history,
                input_messages_key="question",
                history_messages_key="history"
            )
            return retriever, rewrite_with_history, chain_with_history
        except Exception as e:
            print(f"Error during setup_chain: {e}")
            raise
    
    def get_session_history(self, session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = InMemoryChatMessageHistory()
        return self.store[session_id]

    def get_response(self, user_query, received_time, pasted_flag):
        try:
            rewrite_response = self.rewrite_with_history.invoke(
                {"question": user_query},
                config={"configurable": {"session_id": self.session_id}}
            )
            print(rewrite_response)
            rewrite_query = rewrite_response['text']

            try:
                json_str_match = re.search(r'{[\s\S]*}', rewrite_query)
                if json_str_match:
                    json_str = json_str_match.group(0)
                    rewrite_query = json.loads(json_str)['Completed Question']

            except Exception as e:
                print(rewrite_query, e)

            print(rewrite_query)

            # retriever
            contents = self.retriever.invoke(rewrite_query)
            contents = "\n".join([content.page_content for content in contents])
            print(contents)


            response_time = datetime.utcnow().isoformat()

            stored_messages = self.store[self.session_id].messages
            self.store[self.session_id].clear()

            for message in stored_messages[:-1]:
                self.store[self.session_id].add_message(message)
            self.store[self.session_id].add_message(contents)

            response = contents.replace('\n', '\n\n')

            self.save_history_to_pickle()
            self.save_history_to_csv(user_query, rewrite_query, contents, '', received_time, response_time, pasted_flag)

            return response
        except ValueError as ve:
            print(f"ValueError encountered: {ve}")
        except Exception as e:
            print(f"General error encountered: {e}")
            raise
    
    def get_history(self):
        return self.store[self.session_id]
    
    def save_history_to_pickle(self):

        file_path = os.path.join('conversation_history', f"{self.session_id}_history.pkl")
        with open(file_path, "wb") as f:
            pickle.dump(self.store[self.session_id], f)
    
    def save_history_to_csv(self, user_query, rewrite_query, contents, response, received_time, response_time, pasted_flag):

        with csv_lock:
            with open(CSV_FILE, mode='a', encoding='utf-8', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    self.session_id,
                    user_query,
                    rewrite_query,
                    contents,
                    response,
                    received_time,
                    response_time,
                    pasted_flag
                ])
            	


@app.route("/ChatHistory", methods=["GET"])
def get_chatdata():
    session_id = request.args.get("session_id")

    data = []
    with csv_lock:
        with open(CSV_FILE, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if session_id:
                    if row['用户ID'] == session_id:
                        data.append(row)
                else:
                    data.append(row)
    
    if session_id and not data:
        return jsonify({"error": "No data found for the specified session_id."}), 404

    return jsonify({"data": data})


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        user_query = request.form.get("query")
        session_id = request.form.get("prolificID")
        pasted_flag = request.form.get("pasted")

        if user_query:
            received_time = datetime.utcnow().isoformat()
            
            # Check if there's an existing chatbot instance for this session
            if session_id not in chatbot_instances:
                chatbot_instances[session_id] = BasicChatbot(session_id)
            
            chatbot = chatbot_instances[session_id]
            response = chatbot.get_response(user_query, received_time, pasted_flag)
            return jsonify({"response": response})
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=False)#, port=30716)


# export FLASK_APP=flask_app.py
# flask run