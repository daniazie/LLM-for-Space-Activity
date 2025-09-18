from langchain.chat_models import init_chat_model
from langchain_core.tools import tool, StructuredTool
from langchain_core.prompts import ChatMessagePromptTemplate, ChatPromptTemplate
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.runnables import RunnableLambda
from mistralai import Mistral
from dotenv import load_dotenv
from langgraph.store.memory import InMemoryStore
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypedDict
from langgraph.prebuilt import create_react_agent, ToolNode
from langgraph.graph import StateGraph, START, END, MessagesState
from inspect import signature
import os