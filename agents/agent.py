from langchain.chat_models import init_chat_model
from langchain_core.tools import tool, StructuredTool
from langchain_core.prompts import ChatMessagePromptTemplate, ChatPromptTemplate
from mistralai import Mistral
from dotenv import load_dotenv
from langgraph.store.memory import InMemoryStore
from typing import Optional, List, Dict
from typing_extensions import Annotated, TypedDict
from langgraph.prebuilt import create_react_agent, ToolNode
from langgraph.graph import StateGraph, END, MessagesState
from agents.agent_tools_v0 import PreprocessingTool, PDFAgentState, PaperContent, PaperInfo
import os

tools = PreprocessingTool.get_tools()

llm = init_chat_model('mistral-large-latest', model_provider='mistralai')



"""class ExtractKeywords(TypedDict):
    keywords: Annotated[str, ..., 'Keywords or concepts listed by the authors (under Unified Astronomy Thesaurus Concepts, Keywords, etc)']




def data_agent(self):
    data_agent_tools = DataAgentTools()
    get_text_prompt = ChatPromptTemplate.from_messages(
        [
            ('system', 'Extract the contents of the pdf file given.'),
            ('user', '{file}')
        ]
    )

    llm = init_chat_model('mistral-base-latest', model_provider='mistralai')
    
    agent = create_react_agent(llm, data_agent_tools.tools)

    return agent"""