import os
import click
import torch
import streamlit as st
import pandas as pd
import regex as re
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from run_localLLM import run_main_process
from run_localLLM import chatWithLLM

device_type = "cpu"
show_sources = False 
use_history = False 
model_type = "llama" 
save_qa = False 

st.set_page_config(page_title="Companies Information Bot", page_icon="🌐")
st.title("💬 Companies Info Chat Bot")

# Setup memory for contextual conversation
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)

@st.cache_resource(ttl="1h")

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        # Workaround to prevent showing the rephrased question as output
        if prompts[0].startswith("Human"):
            self.run_id_ignore_token = kwargs.get("run_id")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.run_id_ignore_token == kwargs.get("run_id", False):
            return
        self.text += token
        self.container.markdown(self.text)

class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.status = container.status("Company Information Context")

    def on_retriever_start(self, serialized: dict, query: str, **kwargs):
        print("Query :- ", query)
        #self.status.write(f"**Question:** {query}")
        #self.status.update(label=f"**Context Retrieval:** {query}")
        self.status.update(label=f"Company Information Context")

    def on_retriever_end(self, documents, **kwargs):
        for idx, doc in enumerate(documents):
            source = os.path.basename(doc.metadata["source"])
            self.status.write(f"**Document {idx+1} from {source}**")
            self.status.markdown(doc.page_content)
        self.status.update(state="complete")


def clear_content():
    msgs.clear()
    initialMessage = 'Ask me anything about "' + st.session_state['issue'] + '" !'
    msgs.add_ai_message(initialMessage)

with st.sidebar:
    # Dropdown Menu
    option = st.selectbox(
      'Select a company about which you want to know',
      (pd.read_csv(f'data/Company_Names_with_Company_Numbers(Options).csv')),index=None, placeholder="Select a company",on_change=clear_content, key = 'issue')

    # st.write('You selected:', option)
    "[View the source code](https://github.com/TheDataCity/local_LLM)"
    # if tempOption != option:
    #     checkFlag = True
    #     tempOption = option

    show_sources = st.checkbox('Show the Sources')
    use_history = st.checkbox('Use the History')
    save_qa = st.checkbox('Save the Chat') 

    device_type = st.selectbox(
      'Select a device type',
      ("cpu","cuda","ipu","xpu","mkldnn","opengl","opencl","ideep","hip","ve","fpga","ort","xla","lazy","vulkan","mps","meta","hpu","mtia")
      ,index=None, placeholder="Select a device")
    
    model_type = st.selectbox(
      'Select a model type',
      ("llama", "mistral", "non_llama")
      ,index=None, placeholder="Select a LLM model")

    if option is not None:
        last_parenthesis_pos = option.rfind('(')
        # Extracting the text inside the last set of parentheses
        if last_parenthesis_pos != -1:
            companyNumber = st.session_state['issue'][last_parenthesis_pos + 1:-1]  # Slicing from '(' to the end, excluding ')'
        else:
            companyNumber = ""
    
        qa = run_main_process(device_type,show_sources,use_history,model_type,save_qa, company_number=companyNumber)
        chackFlag = False

if not option:
        st.info("Please select a company from the drop down menu")
        st.stop()

avatars = {"human": "❓", "ai": "❄️"}
for msg in msgs.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)

if user_query := st.chat_input(placeholder="Ask me anything about the company!"):
    st.chat_message("user", avatar= "❓").write(user_query)

    with st.chat_message("assistant", avatar="❄️"):
        print("user_query :- ", user_query)
        response, sources = chatWithLLM(show_sources, save_qa, qa, user_query)
        st.write(response)
        print(sources)
        # response = qa_chain.run(user_query, callbacks=[retrieval_handler, stream_handler])
        # st.json(jsonResponse)
