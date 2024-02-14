import time
import os
import random
import gradio as gr
from pathlib import Path
from openai import OpenAI as OAI
from langchain_openai import OpenAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings


try:
    print("Connecting to Server...")
    client = OAI(base_url="http://localhost:1234/v1", api_key="not-needed")
except:
    print("Couldn't connect to Server")


embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})

default_prompt = "You are an intelligent assistant. You always provide well-reasoned answers that are both correct and helpful. Only answer the asked questions"
history = [
    {"role": "system", "content": default_prompt},
]

doc_name = "cam.pdf"
vector_name = f"{Path(doc_name).stem}"

def text_split(doc_name):
    
    loaders = [PyPDFLoader(doc_name)]
    docs = []
    for file in loaders:
        docs.extend(file.load())
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(docs)
    
    return docs

def vectoring(vector_name):
    
    if os.path.isdir(vector_name):
        print("Splitting Document")
        docs = text_split(doc_name)
        print("VectorDB Found")
        vector_db = Chroma(persist_directory=vector_name, embedding_function=embedding_function)
        print("Number of DB Collections:", vector_db._collection.count())
        return vector_db

    else:
        print("Creating Folder:", vector_name)
        os.makedirs(vector_name)
        print("Splitting Document")
        docs = text_split(doc_name)
        print("Creating VectorDB...")
        vector_db = Chroma.from_documents(docs, embedding_function, persist_directory=vector_name)

        print("Number of DB Collections:", vector_db._collection.count())
        return vector_db
        
vector_db = vectoring(vector_name)

def respond(message, func_history):
    search_results = vector_db.similarity_search(message, k=2)
    some_context = ""
    for result in search_results:
        some_context += result.page_content + "\n\n"
        
    history.append({"role": "user", "content": some_context + message})
    
    completion = client.chat.completions.create(
        model="local-model", 
        messages=history,
        temperature=0.7,
        stream=True,
    )

    new_message = {"role": "assistant", "content": ""}
    
    for chunk in completion:
        if chunk.choices[0].delta.content:
            new_message["content"] += chunk.choices[0].delta.content

    history.append(new_message)
    func_history.append((message, new_message["content"]))
    return "", func_history

def change_system_prompt(new_system_prompt):
    history[0]["content"] = new_system_prompt
    return "New Prompt Saved!"


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            system_prompter = gr.Textbox(label="System Prompt", placeholder=default_prompt)
            system_prompter_outcome = gr.Textbox()
        btn = gr.Button("Apply New Prompt")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder= "Tell me about the text")
    clear = gr.ClearButton([msg, chatbot])

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    btn.click(fn=change_system_prompt, inputs=system_prompter, outputs=system_prompter_outcome)

demo.launch()
