import os
from azure.search.documents.models import VectorizedQuery
from azure.search.documents import SearchClient
from numpy import index_exp
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
import re
from openai import AzureOpenAI
import os
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery
from azure.search.documents import SearchClient
import pandas as pd
import json
from datetime import datetime
from utils.UsefulFunctions import log_query_to_blob, format_llm_response

load_dotenv()

# Stage 1: classifiy the question into framework recommender, about the framework and out of scope                                                                                                                                                      
def invoke_llm_stage_1(user_query: str):
    client = AzureOpenAI(api_key=os.getenv("openai_api_key"),
                         api_version=os.getenv("openai_api_version"),
                         azure_endpoint=os.getenv("openai_azure_endpoint"))
    
    system_Prompt = """
    Classify the query into these categories:

    - 'recommendation': If the user is asking for framework or agreement or service or services recommendations. 
    - 'details':  If the user is asking about a specific framework or frame work.
    - 'out_of_scope': If the question is beyond the provided frameworks. 
    
    Return only one category.
    """

    message_text = [{'role':'system', 'content':system_Prompt},
                    {'role':'user', 'content': user_query}]
    
    response = client.chat.completions.create(model='gpt-4o-code',
                                              messages=message_text).choices[0].message.content
    # print(response)
    
    return response

# Define state
class State(TypedDict):
    query: str
    query_classification: str
    framework_recommder: str
    framework_details: str
    output: str
    framework_numbers: list
    # query_out_of_scope: str
    query_embeddings: list
    Framework_list: list
    Framework_classification: str
    FD_query_embeddings: list
    Framework_Details_list: list

# LLM-baser stage 1 Router mode 
def llm_call_router_stage_1(state: State):
    
    query_classification = invoke_llm_stage_1(state['query'])
    # # print(query_classification)
    return {'query_classification': query_classification.strip().lower()}

# framework recommender Node : get embeddings 
def FR_step_1_Query_Embeddings(state: State):

    client = AzureOpenAI(api_key=os.getenv("openai_api_key"),
                         api_version=os.getenv("openai_api_version"),
                         azure_endpoint=os.getenv("openai_azure_endpoint"))
    embed = client.embeddings.create(model="text-embedding-ada-002", 
                                     input=state['query']).data[0].embedding
    
    return {"query_embeddings": embed}
    # return {"output": 'you have guided to the recommendation system. '}


def FR_step_2_Search_Chuncks(state: State):

    search_client = SearchClient(endpoint=os.getenv('azure_search_service_endpoint'),
                                index_name=os.getenv('azure_index_FM_recommender_name'), 
                                credential=AzureKeyCredential(os.getenv('azure_search_api_key')))

    query_embeddings = state['query_embeddings']
    vector_query = VectorizedQuery(vector=query_embeddings,
                                   k_nearest_neighbors=20,
                                   fields="embeddings")
    search_results = search_client.search(search_text=None,
                                          vector_queries=[vector_query],
                                          select=['frameworknumber', 'framework', 'frameworkdescchunk'])
    
    results = []
    for result in search_results:
        if result['@search.score']>0.8:
            results.append({'frameworkdescchunk' : result['frameworkdescchunk']})
    
    return {"Framework_list": results}

def FR_step_3_LLM(state: State):

    client = AzureOpenAI(api_key=os.getenv("openai_api_key"),
                    api_version=os.getenv("openai_api_version"),
                    azure_endpoint=os.getenv("openai_azure_endpoint"))

    system_prompt = "You are a chatbot expert in the CCS framework recommender. your task is to for mat the question and provided search results into a clear, concise, and well-structured answer that directly addresses the question."
    user_prompt = f"Question: {state['query']}\n\nSearch Results:\n{state['Framework_list']}\n\nFormat the question and search results into a summary that answers the question:"

    messsage_text = [{'role':'system', 'content': system_prompt},
                     {'role':'user', 'content':user_prompt}]
    

    response = client.chat.completions.create(model="gpt-4o-code",
                                              messages=messsage_text).choices[0].message.content

    return {"output": response, 
            "framework_numbers": re.findall(r'RM\d+', response)}

# out of scope:
def query_out_of_scope(state: State):
    query = state["query"]
    
    response_message = f"""
    <p>Your question '<strong>{query}</strong>' is beyond our scope.</p>

    <p>Please refine your question to be about framework recommendations or details.</p>

    <p>Example of valid questions:</p>
    <ul>
        <li>"Which framework is best for digital services?"</li>
        <li>"Tell me about RM6098 and how to buy it."</li>
    </ul>

    <p>Please rewrite your question accordingly:</p>
    """
    
    # Simulating human intervention by waiting for input
    # new_query = input(response_message)  # Ask user for new input
    
    return {"output": response_message, 
            "framework_numbers": []}  # Returns the new query for reprocessing


def route_decision_stage_1(state: State):
    # Get the raw classification and normalize it
    query_classification = state.get("query_classification", "").strip().lower()
    
    # Add debugging to see the exact value
    # # print(f"DEBUG - Classification received: '{query_classification}'")
    
    # Check for various forms the classification might take
    if "recommendation" in query_classification:
        return "FR_step_1_Query_Embeddings"
    elif "details" in query_classification:
        return "Framework_Supervisor"
    elif "out" in query_classification and "scope" in query_classification:
        # print('Out of scope is found.')
        return "query_out_of_scope"
    else: 
        # As a fallback, you could default to a specific path
        # print(f"WARNING: Unexpected classification: '{query_classification}'. Defaulting to out_of_scope.")
        return "query_out_of_scope"
    
# Stage 2
# Frameworknumber extraction. 
def invoke_FramworkNumber_extraction_llm_stage_2(user_query: str):
    # Initialize the Azure OpenAI client
    client = AzureOpenAI(
        api_key=os.getenv("openai_api_key"),
        api_version=os.getenv("openai_api_version"),
        azure_endpoint=os.getenv("openai_azure_endpoint")
    )

    # Define the framework mapping
    frameworks = {
        "RM1043.8": "Digital Outcomes 6",
        "RM6098": "Technology Products & Associated Services 2",
        "RM6187": "Management Consultancy Framework Three (MCF3)",
        "RM6116": "Network Services 3",
        "RM6264": "Facilities Management and Workplace Services DPS"
    }

    # Create the system and user prompts
    system_prompt = (
        "You are an expert in identifying the framework details from the query. "
        f"Here are the list of framework numbers and their names: {frameworks}"
    )
    
    user_prompt = (
        f"This is the customer query: {user_query}. "
        f"For your reference, please use the following framework information: {frameworks}. "
        "Can you only generate the framework number or 'No-Framework'?"
    )

    # Prepare the messages for the API call
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}]

    # Make the API call to Azure OpenAI
    response = client.chat.completions.create(model="gpt-4o-code",
                                              messages=messages).choices[0].message.content
    
    # print(response)
    
    valid_frameworks = ["rm1043.8", "rm6098", "rm6187", "rm6116", "rm6264"]

    for framework in valid_frameworks:
        if framework in response.lower():
            return framework
        

    # Extract and return the framework number or "No-Framework"
    return "No-Framework"

def llm_call_router_stage_2(state: State):
    Framework_classification = invoke_FramworkNumber_extraction_llm_stage_2(state['query'])
    # print(Framework_classification)
    return {'Framework_classification': Framework_classification.strip().lower()}

def route_decision_stage_2(state: State):
    Framework_classification = state.get("Framework_classification", "").strip().lower()
    # print(Framework_classification)

    if Framework_classification.upper() in ["RM1043.8", "RM6098", "RM6187", "RM6116", "RM6264"]:
        return "FD_step_1_Query_Embeddings"
    elif Framework_classification.lower() == "No-Framework".lower():
        return "Framework_requery"
    else: 
        raise ValueError(f"Unexpected routing decision: {Framework_classification}")


# framework recommender Node : get embeddings 
def FD_step_1_Query_Embeddings(state: State):

    client = AzureOpenAI(api_key=os.getenv("openai_api_key"),
                         api_version=os.getenv("openai_api_version"),
                         azure_endpoint=os.getenv("openai_azure_endpoint"))
    embed = client.embeddings.create(model="text-embedding-ada-002", 
                                     input=state['query']).data[0].embedding
    
    return {"FD_query_embeddings": embed}


def FD_step_2_Search_Chuncks(state: State):

    Framework_classification = state['Framework_classification']
    index_names = {'rm1043.8':'azd-uks-ai-rm10438',
                   "rm6098": "azd-uks-ai-rm6098",
                   "rm6187": "azd-uks-ai-rm6187",
                   "rm6116": "azd-uks-ai-rm6116",
                   "rm6264": "azd-uks-ai-rm6264"}
    # print(index_names[Framework_classification])
    search_client = SearchClient(endpoint=os.getenv('azure_search_service_new_endpoint'),
                                index_name=index_names[Framework_classification], 
                                credential=AzureKeyCredential(os.getenv('azure_search_new_api_key')))

    query_embeddings = state['FD_query_embeddings']
    vector_query = VectorizedQuery(vector=query_embeddings,
                                   k_nearest_neighbors=20,
                                   fields="embeddings")
    search_results = search_client.search(search_text=None,
                                          vector_queries=[vector_query],
                                          select=['Framework', 'Filename', 'content'])
    
    results = []
    for result in search_results:
        if result['@search.score']>0.8:
            results.append({'content' : result['content'], 
                            'references':result['Filename']})
    
    return {"Framework_Details_list": results}

def FD_step_3_LLM(state: State):

    client = AzureOpenAI(api_key=os.getenv("openai_api_key"),
                    api_version=os.getenv("openai_api_version"),
                    azure_endpoint=os.getenv("openai_azure_endpoint"))

    system_prompt = "You are a chatbot expert in the CCS framework recommender. Your task is to generate a detailed summary based on the user’s query and provided content chunks. The summary should be well-structured, concise, and directly answer the query. you also offer suggestions supported by credible citations"
    user_prompt = f"Question: {state['query']}\n\nSearch Results:\n{state['Framework_Details_list']}\n\nCreate a detailed summary of the query based on the provided content chunks with citations. Ensure the summary is clear, structured, and directly addresses the query."


    messsage_text = [{'role':'system', 'content': system_prompt},
                     {'role':'user', 'content':user_prompt}]    

    response = client.chat.completions.create(model="gpt-4o-code",
                                              messages=messsage_text).choices[0].message.content

    return {"output": response,
            "framework_numbers": state['Framework_classification'].upper()}


def Framework_requery(state: State):
    query = state["query"]
    
    response_message = f"""
    Your question <b>{query}</b> is beyond our scope.

    This prototype can only address inquiries related to the following five frameworks:
    <ul>
    <li><b>RM1043.8</b> (Digital Specialists and Programmes)</li>
    <li><b>RM6098</b> (Technology Products & Associated Services 2)</li>
    <li><b>RM6187</b> (Management Consultancy Framework Three - MCF3)</li>
    <li><b>RM6116</b> (Network Services 3)</li>
    <li><b>RM6264</b> (Facilities Management and Workplace Services DPS)</li>
    </ul>
    Valid examples of questions include:
    <ul>
    <li>In RM6098, what guidelines must suppliers follow during the standstill period before finalizing a contract award under the framework?</li>
    <li>Tell me about RM6098 and how to buy it.</li>
    </ul>
    """
    
    return {"output": response_message,
            "framework_numbers": []}  # Returns the new query for reprocessing

# Build the workflow using langgraph
workflow = StateGraph(State)

# Add nodes
workflow.add_node('Supervisor', llm_call_router_stage_1)
workflow.add_node('FR_step_1_Query_Embeddings', FR_step_1_Query_Embeddings)
workflow.add_node("FR_step_2_Search_Chuncks", FR_step_2_Search_Chuncks)
workflow.add_node("FR_step_3_LLM", FR_step_3_LLM)
workflow.add_node('query_out_of_scope', query_out_of_scope)
workflow.add_node('Framework_Supervisor', llm_call_router_stage_2)
workflow.add_node('Framework_requery', Framework_requery)
workflow.add_node('FD_step_1_Query_Embeddings', FD_step_1_Query_Embeddings)
workflow.add_node('FD_step_2_Search_Chuncks', FD_step_2_Search_Chuncks)
workflow.add_node('FD_step_3_LLM', FD_step_3_LLM)

# add edges 
workflow.add_edge("FR_step_1_Query_Embeddings", "FR_step_2_Search_Chuncks")
workflow.add_edge("FR_step_2_Search_Chuncks", "FR_step_3_LLM")
workflow.add_edge("FR_step_3_LLM", END)
workflow.add_edge("FD_step_1_Query_Embeddings", 'FD_step_2_Search_Chuncks')
workflow.add_edge("FD_step_2_Search_Chuncks", 'FD_step_3_LLM')
workflow.add_edge("FD_step_3_LLM", END)
workflow.add_edge("Framework_requery", END)
workflow.add_edge("query_out_of_scope", END)


# Add conditional routing 
workflow.add_conditional_edges("Supervisor", route_decision_stage_1, {"FR_step_1_Query_Embeddings":"FR_step_1_Query_Embeddings",
                                                                                   "Framework_Supervisor": "Framework_Supervisor",
                                                                                   "query_out_of_scope":"query_out_of_scope"})

workflow.add_conditional_edges("Framework_Supervisor", route_decision_stage_2, {"FD_step_1_Query_Embeddings":"FD_step_1_Query_Embeddings",
                                                                                   "Framework_requery": "Framework_requery"})

# set the entry point 
workflow.set_entry_point("Supervisor")

# compile the workflow
Framwork_app = workflow.compile()

# --------------------------- Main app -----------------------------------------

def MultiAgent_Answering(query):
    state = Framwork_app.invoke({"query": query})
    answer = state["output"]
    framework_numbers =state["framework_numbers"]
    log_query_to_blob(query, answer)
    return answer, framework_numbers
