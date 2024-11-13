import os
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from langchain_ibm import WatsonxEmbeddings, WatsonxLLM
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenTextParamsMetaNames
from langchain_elasticsearch import ElasticsearchStore
from langchain_community.tools.tavily_search import TavilySearchResults
from crag.agent import RagAgent,RagAgentInput
from langchain_core.documents import Document

load_dotenv()

def run():
    # watsonx.ai credentials
    credentials = {
        "url": "https://us-south.ml.cloud.ibm.com",
        "apikey": os.getenv("GENAI_KEY"),
        "projectId": os.getenv("GENAI_PROJECT_ID")
    }

    # Get the vector store that serves as our knowledge base
    vectorstore = getStore(credentials,indexName="cascon_1")
    # Next, we need to define the Retriever.
    retriever = vectorstore.as_retriever()
    # Define the web search tool
    web_search_tool = TavilySearchResults()
    grader_llm = getLLM(credentials,model_id="meta-llama/llama-3-405b-instruct")
    llm = getLLM(credentials,model_id="meta-llama/llama-3-405b-instruct")
    # Next, we define a RAG chain using LangChain.
    # Now, we're getting to the good stuff. Let's define our Graph class and the 'nodes' in our corrective RAG graph. These 'nodes', defined as python functions, correspond to the blocks in the diagrams at the beggining of this tutorial.
    rag_agent_input = RagAgentInput(
       agent_name = "cascon",
       retriever = retriever,
       web_search_tool = web_search_tool,
       query_decompose_llm = llm,
       grader_llm = grader_llm,
       rag_llm = llm
    )
    rag_agent = RagAgent(rag_agent_input)
    rag_agent.saveGraphImage()
    question = "Which David Fincher film that stars Edward Norton does not star Brad Pitt?"
    answer = rag_agent.query(question)
    print(answer)
    

    
def getStore(credentials,indexName: str) -> ElasticsearchStore:
    #elasticsearch api keys
    url = os.getenv("WXD_URL", None)
    username= os.getenv("WXD_USERNAME", None)
    password = os.getenv("WXD_PASSWORD", None)
    
    #initialize elasticsearch client
    es_client = Elasticsearch(
        url,
        basic_auth=(username,password),
        verify_certs=False,
        request_timeout=3600)

    #define embeddings model
    embeddings = WatsonxEmbeddings(model_id='ibm/slate-125m-english-rtrvr',
                               apikey=credentials.get('apikey'),
                               url=credentials.get('url'),
                               project_id=credentials.get("projectId"))
    
    #load existing vectorstore
    vectorstore = ElasticsearchStore(
     es_connection=es_client,
     embedding = embeddings,
     index_name=indexName)
    return vectorstore

def getLLM(credentials, model_id: str):
    llm = WatsonxLLM(
        model_id = model_id,
        url=credentials.get("url"),
        apikey=credentials.get("apikey"),
        project_id=credentials.get("projectId"),
        params = {  GenTextParamsMetaNames.DECODING_METHOD: "greedy",
                GenTextParamsMetaNames.MAX_NEW_TOKENS: 200,
                GenTextParamsMetaNames.MIN_NEW_TOKENS: 10})
    return llm

