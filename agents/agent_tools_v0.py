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
from langgraph.graph import StateGraph, END, MessagesState
from inspect import signature
import os

class PaperInfo(TypedDict):
    title: Annotated[str, ..., 'Title of the research paper']
    author: Annotated[str, ..., 'Authors of the paper in (Name) et al. format']
    id: Annotated[str, ..., 'DOI or arXiv ID']
    data: Annotated[List[str], ..., 'Data used']
    code_link: Annotated[str, 'None provided', 'Link to code (Github, etc)']
    packages_used: Annotated[List[str], 'None Provided', 'Packages or libraries used.']
    task: Annotated[str, ..., 'Task performed in paper (e.g. flare prediction, CME detection, etc.)']
    field: Annotated[str, ...,'What branch of Space Science/Solar Physics e.g. solar flares, CME, solar wind, etc.']
    abstract: Annotated[str, ..., 'Summary of abstract']
    models: Annotated[List[str], ..., 'Model architecture(s) used e.g. CNN, ResNet, pix2pix, GAN, etc.']
    hybrid_model: Annotated[bool, ..., 'Whether hybrid model architectures were used.']
    multimodal: Annotated[List[str], 'N/A', 'Type of multimodal models used, if any.']
    baselines: Annotated[List[str], 'N/A', 'Baseline models or papers used, along with citations in the format: baseline (author, year).']
    preprocessing: Annotated[List[str], ..., 'Preprocessing steps taken']
    citations: Annotated[List[str], ..., 'Citations']
    approach_used: Annotated[List[str], ..., 'Approach(es) used']
    methodology: Annotated[str, ..., 'Summary of methodology']
    metrics: Annotated[List[Dict[str, float]], ..., 'Metrics used for evaluation and the scores obtained']
    limitations: Annotated[List[str], ..., 'Limitations of the research']
    reproducibility: Annotated[float, ..., 'From 0-5, based on the completeness of the method, code and data, how reproducible is the paper?']
    strengths: Annotated[str, ..., 'Key strengths']
    weaknesses: Annotated[str, ..., 'Key weaknesses']
    reuse_potential: Annotated[List[str], 'N/A', 'Notes on potential for reuse (if applicable)']

class FileInput(BaseModel):
    file: str = Field(description='Path to PDF file')

class PaperInput(BaseModel):
    paper: str = Field(description='Research paper.')

class PaperContent(BaseModel):
    title: str = Field(description='Title of pdf')
    content: str = Field(description='Content of pdf')

class PDFAgentState(MessagesState):
    final_response: PaperContent

class PreprocessingTool:
    def __init__(self):
        load_dotenv()
        self.client = Mistral(api_key=os.environ['MISTRAL_API_KEY'])
        self.llm = init_chat_model('mistral-large-latest', model_provider='mistralai')

    def _run(
        self,
        *args: Any,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
        ) -> Any:
        """Use the tool."""
        if self.func:
            new_argument_supported = signature(self.func).parameters.get("callbacks")
            bound_func = self.func.__get__(self, self.__class__)  # Bind the function to the instance
            return (
                bound_func(
                    *args,
                    callbacks=run_manager.get_child() if run_manager else None,
                    **kwargs,
                )
                if new_argument_supported
                else bound_func(*args, **kwargs)
            )
        raise NotImplementedError("Tool does not support sync")
    

    
    def get_tools(self, file=None, paper=None):
        def upload_pdf(file):
            """Upload pdf file for OCR.
            
            Args:
                file: Files
            """
            pdf = self.client.files.upload(
                file={
                    'file_name': file,
                    'content': open(file, 'rb')
                },
                purpose='ocr'
            )

            signed_url = self.client.files.get_signed_url(file_id=pdf.id)
            return signed_url.url
    
        def extract_pdf(file):
            """Extract texts from a pdf
            
            Args:
                file: File to be processed"""
            
            title = file.split('.pdf')[0].strip()
            title = title.split('/')[-1].strip() if '/' in title else title

            ocr_response = self.client.ocr.process(
                model='mistral-ocr-latest',
                document={
                    'type': 'document_url',
                    'document_url': upload_pdf(file)
                },
                include_image_base64=True
            )

            content = []
            for page in ocr_response.pages:
                content.append(page.markdown)

            return '\n'.join(content)
         
        def preprocessing(paper: str):
            """Clean the output of the extracted information
            
            Args:
                paper: Extracted paper to be reorganised.
            """
            paper['metrics_list'] = paper['metrics'].copy()
            paper['metrics'] = []
            paper['scores'] = []

            for i in range(len(paper['metrics_list'])):
                for metric, score in paper['metrics_list'][i].items():
                    paper['metrics'].append(metric)
                    paper['scores'].append(score)

            return dict(paper)

        extract_prompt = ChatPromptTemplate.from_messages([
            ('system', "Extract information from the following research paper."),
            ('human', '{paper}')  # This expects {"paper": "text"}
        ])
        
        structured_llm = self.llm.with_structured_output(PaperInfo)
        preprocessing_run = RunnableLambda(preprocessing)

        chain = extract_prompt | structured_llm # | preprocessing_run

        extract_info_tool = chain.as_tool(
            name="InformationExtractor",
            description="Extract key information from research papers.",
            args_schema=PaperInput,
        )
        
        extract_pdf_tool = StructuredTool.from_function(
            func=extract_pdf,
            name='PDFExtractor',
            description='Extract content from PDF',
            args_schema=FileInput,
            return_direct=True,
        )

        return [extract_pdf_tool, extract_info_tool]