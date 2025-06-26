# importing the important libraries
import os
import re
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime
import json

from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

from utils.UsefulFunctions import reading_prompt, log_query_to_blob_v2

load_dotenv()

def get_Embeddings(query:str):

    client = AzureOpenAI(api_key=os.getenv("openai_api_key"),
                         api_version=os.getenv("openai_api_version"),
                         azure_endpoint=os.getenv("openai_azure_endpoint"))
    embed = client.embeddings.create(model="text-embedding-ada-002", 
                                     input=query).data[0].embedding
    
    return embed

def search_index(query:str, classification:str, index_name:str, endpoint:str, framework:str = None, top:int = 5):

    search_client = SearchClient(endpoint=endpoint,
                                index_name=index_name, 
                                credential=AzureKeyCredential(os.getenv('azure_search_new_api_key')))
    
    # Build the filter expression
    filter_expr = f"Classification eq '{classification}'"
    
    if framework:
        filter_expr += f" and Framework eq '{framework}'"

    search_embeddings = get_Embeddings(query)    

    # Execute pure vector search
    results = search_client.search(
        search_text="",  # Empty string for no full-text search
        vector_queries=[
            VectorizedQuery(
                vector=search_embeddings,
                k_nearest_neighbors=top,
                fields="embeddings",
                exhaustive=True
            )
        ],
        filter=filter_expr,
        top=top,
        select=["id", "content", "Type", "Framework", "Filename", "Framework_link", "Classification"]
    )

    results_Content = []

    for result in results:
        if result['@search.score']>0.8:
            results_Content.append({'content' : result['content'], 
                                    'Framework_link': result['Framework_link'],
                                    "Classification": result["Classification"],
                                    "Filename": result["Filename"],
                                    "Framework": result["Framework"]})

    return results_Content

def system_prompt(classification: str):

    system_prompt_filename = classification + '_system_prompt.txt'
    system_prompt = reading_prompt(system_prompt_filename)
    
    return system_prompt


def user_prompt(classification: str, results: list, query_text:str):
    user_prompt_filename = classification + '_user_prompt.txt'
    user_prompt = reading_prompt(user_prompt_filename)

    # Initialize reterive_chunk with empty string
    reterive_chunk = ''
    
    if results:
        results_df = pd.DataFrame(results)
        results_chunks = results_df['content']

        for i in range(len(results_chunks)):
            chunk_text = results_df.loc[i, 'content']
            fr_no = results_df.loc[i, 'Framework']
            if fr_no:
                # print(fr_no)
                reterive_chunk += chunk_text + '. Framework Number is '+ fr_no+ '.\n'
            else: 
                reterive_chunk += chunk_text + '.\n'

    user_prompt = user_prompt.replace('{query_text}', query_text)
    user_prompt = user_prompt.replace('{retrieved_chunks}', reterive_chunk)

    return user_prompt

def ResponseFromLLM(system_prompt:str, user_prompt:str):

    client = AzureOpenAI(api_key=os.getenv("openai_api_key"),
                         api_version=os.getenv("openai_api_version"),
                         azure_endpoint=os.getenv("openai_azure_endpoint"))    
    
    message_text = [{'role':'system', 'content':system_prompt},
                    {'role':'user', 'content': user_prompt}]
    
    response = client.chat.completions.create(model='gpt-4o-code',
                                              messages=message_text).choices[0].message.content
    
    return response

def get_framework_list(response: str):
    
    return response

def invoke_llm_stage_1(user_query: str):
    client = AzureOpenAI(api_key=os.getenv("openai_api_key"),
                         api_version=os.getenv("openai_api_version"),
                         azure_endpoint=os.getenv("openai_azure_endpoint"))
    
    system_Prompt = """
    Classify the user's query into exactly one of these categories based on CCS procurement frameworks:

    1. 'details': Questions about specific frameworks/agreements (e.g., "Where can I find RM6098?", "What's the procedure for Network Services 3?")
    2. 'recommendation': Requests for framework suggestions (e.g., "Which framework for office furniture?", "What agreement for temporary medical staff?", or standalone terms like "Digital" or "Cloud" that match framework categories)
    3. 'comparison': Direct comparisons between frameworks (e.g., "RM6098 vs RM6116", "difference between RM1043.8 and RM6098")
    4. 'startup': New user questions about getting started (e.g., "I'm new to CCS", "first step to using frameworks", "onboarding guide")
    5. 'not scope': Questions unrelated to framework procurement (e.g., "increase text size", "save previous searches", "website feedback")

    Key indicators:
    - If the query matches or partially matches any of these framework categories (even partially), classify as 'recommendation': Cloud and Hosting, Construction, Digital and Technology Services, Energy, Facilities Management, Financial Services, Fleet, Hardware, HR and Workforce Services, Low Value, Network Services, Outsourced Services, Professional Services, Software, Travel, Accommodation and Venues
    - Specific framework codes (RMXXXX) usually indicate 'details' or 'comparison'
    - "Which framework for..." typically indicates 'recommendation'
    - New user terminology ("new to", "get started", "first steps") indicates 'startup'
    - Technical questions about the website itself are 'not scope'

    Return only the category name in lowercase, nothing else.
    """

    message_text = [{'role':'system', 'content':system_Prompt},
                    {'role':'user', 'content': user_query}]
    
    response = client.chat.completions.create(model='gpt-4o-code',
                                              messages=message_text).choices[0].message.content
    
    return response

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

##------------------------------------------------------------------------------------------------------------
class State(TypedDict):
    query: str

    # supervisor outputs
    query_classification: str # details, recommendation, comparison, startup, not scope
    Framework_classification: str

    # filters used in the 
    data_classification: str # 'details' 'startup' 'recommendation'
    frameworkNumber : str # 

    # outputs willbe used in the response
    weblinks: list
    frameworks: list
    filenames: list
    frameworkNames: list
    output: str
    framework_numbers: list
    LLM_output: str
    LLM_2_response: str

# Stage 1: classifiy the question into framework recommender, about the framework and out of scope                                                                                                                                                      


# LLM-baser stage 1 Router mode 
def llm_call_router_stage_1(state: State):
    
    query_classification = invoke_llm_stage_1(state['query'])
    # print('Query classified into : ', query_classification)
    return {'query_classification': query_classification.strip().lower()}

def route_decision_stage_1(state: State):
    # Get the raw classification and normalize it
    query_classification = state.get("query_classification", "").strip().lower()
    
    # Check for various forms the classification might take
    if "recommendation" in query_classification:
        return "Framework_Recommendation_agent"
    elif "details" in query_classification:
        return "Framework_Supervisor"
    elif "out" in query_classification and "scope" in query_classification:
        # print('Out of scope is found.')
        return "query_out_of_scope"
    elif "startup" in query_classification:
        return "Startup_agenet"
    elif "comparison" in query_classification:
        return "Comparison_agent"
    else: 
        # As a fallback, you could default to a specific path
        # print(f"WARNING: Unexpected classification: '{query_classification}'. Defaulting to out_of_scope.")
        return "query_out_of_scope"
    

# framework recommender Node : get embeddings 
def Framework_Recommendation_agent(state: State):
    query = state['query']
    data_classification = 'recommendation'
    query_classification = state['query_classification']

    # search_client = SearchClient(endpoint=os.getenv('azure_search_service_new_endpoint'),
    #                             index_name=os.getenv('azure_index_FM_recommender_name'), 
    #                             credential=AzureKeyCredential(os.getenv('azure_search_new_api_key')))

    # query_embeddings = get_Embeddings(query) 

    # vector_query = VectorizedQuery(vector=query_embeddings,
    #                                k_nearest_neighbors=10,
    #                                fields="embeddings")
    # search_results = search_client.search(search_text=None,
    #                                       vector_queries=[vector_query],
    #                                       select=['frameworknumber', 'framework', 'frameworkdescchunk'])
    
    # results = []
    # for result in search_results:
    #     if result['@search.score']>0.75:
    #         results.append({'content' : result['frameworkdescchunk'],
    #                         'Framework_link': f"https://www.crowncommercial.gov.uk/agreements/{result['frameworknumber']}",
    #                         'Classification': data_classification ,
    #                         "Filename": result["framework"],
    #                         "Framework": result["framework"]})

    query_classification = state['query_classification']

    results = search_index(query, data_classification, os.getenv('azure_index_wholedata'), os.getenv('azure_search_service_new_endpoint'))

    systems_prompt = system_prompt(query_classification)
    users_prompt = user_prompt(query_classification, results, query)

    response = ResponseFromLLM(systems_prompt, users_prompt)
       
    return {"LLM_output": response}

def Startup_agenet(state: State):

    query = state['query']
    data_classification = 'startup'
    query_classification = state['query_classification']

    results = search_index(query, data_classification, os.getenv('azure_index_wholedata'), os.getenv('azure_search_service_new_endpoint'))

    systems_prompt = system_prompt(query_classification)
    users_prompt = user_prompt(query_classification, results, query)

    response = ResponseFromLLM(systems_prompt, users_prompt)
       
    return {"LLM_output": response}

def Comparison_agent(state: State):

    query = state['query']
    data_classification = 'recommendation'
    query_classification = state['query_classification']

    # search_client = SearchClient(endpoint=os.getenv('azure_search_service_endpoint'),
    #                             index_name=os.getenv('azure_index_FM_recommender_name'), 
    #                             credential=AzureKeyCredential(os.getenv('azure_search_api_key')))

    # query_embeddings = get_Embeddings(query) 

    # vector_query = VectorizedQuery(vector=query_embeddings,
    #                                k_nearest_neighbors=20,
    #                                fields="embeddings")
    # search_results = search_client.search(search_text=None,
    #                                       vector_queries=[vector_query],
    #                                       select=['frameworknumber', 'framework', 'frameworkdescchunk'])
    
    # results = []
    # for result in search_results:
    #     if result['@search.score']>0.8:
    #         results.append({'content' : result['frameworkdescchunk'],
    #                         'Framework_link': f"https://www.crowncommercial.gov.uk/agreements/{result['frameworknumber']}",
    #                         'Classification': data_classification ,
    #                         "Filename": result["framework"],
    #                         "Framework": result["framework"]})

    results = search_index(query, data_classification, os.getenv('azure_index_wholedata'), os.getenv('azure_search_service_new_endpoint'))

    systems_prompt = system_prompt(query_classification)
    users_prompt = user_prompt(query_classification, results, query)

    response = ResponseFromLLM(systems_prompt, users_prompt)
       
    return {"LLM_output": response}

def query_out_of_scope(state: State):
    query = state["query"]
    
    response_message = f"""
    <p>Your question '<strong>{query}</strong>' is beyond our scope.</p>
  
  <div class="guidance">
    <p>Please refine your question to be about framework recommendations or details.</p>
    <p>Example of valid questions:</p>
    <ul>
      <li>"Which framework is best for digital services?"</li>
      <li>"Tell me about RM6098 and how to buy it."</li>
    </ul>
    <p>Please rewrite your question accordingly:</p>
  </div>
    """
    
    # Simulating human intervention by waiting for input
    # new_query = input(response_message)  # Ask user for new input
    
    return {"output": response_message,
            "framework_numbers": []}  # Returns the new query for reprocessing

def llm_call_router_stage_2(state: State):
    Framework_classification = invoke_FramworkNumber_extraction_llm_stage_2(state['query'])
    # print(Framework_classification)
    return {'Framework_classification': Framework_classification.strip().lower()}


def route_decision_stage_2(state: State):
    Framework_classification = state['Framework_classification']

    if Framework_classification.upper() in ["RM1043.8", "RM6098", "RM6187", "RM6116", "RM6264"]:
        return "Framework_details_agent"
    elif Framework_classification.lower() == "No-Framework".lower():
        return "Framework_requery"
    else: 
        raise ValueError(f"Unexpected routing decision: {Framework_classification}")
    
def Framework_details_agent(state: State):

    Framework_classification = state['Framework_classification'].upper()
    # print(Framework_classification)
    
    query = state['query']
    data_classification = 'details'
    query_classification = state['query_classification']

    results = search_index(query, data_classification, os.getenv('azure_index_wholedata'), os.getenv('azure_search_service_new_endpoint'), Framework_classification, 5)

    systems_prompt = system_prompt(query_classification)
    users_prompt = user_prompt(query_classification, results, query)

    response = ResponseFromLLM(systems_prompt, users_prompt)
       
    return {"LLM_output": response}

def Framework_requery(state: State):
    query = state["query"]
    
    response_message = f"""
    <p>Your question '<strong>{query}</strong>' is beyond our scope.</p>
  
  <div class="guidance">
    <p>Please refine your question to be about framework recommendations or details.</p>
    <p>Example of valid questions:</p>
    <ul>
      <li>"Which framework is best for digital services?"</li>
      <li>"Tell me about RM6098 and how to buy it."</li>
    </ul>
    <p>Please rewrite your question accordingly:</p>
  </div>
    """
    
    return {"output": response_message,
            "framework_numbers": []}  # Returns the new query for reprocessing

def plaing_english_numbers(state: State):
    LLM_response = state['LLM_output']
    query_text = state['query']

    system_prompt_file = 'plain_english_system_prompt.txt'
    systems_prompts = reading_prompt(system_prompt_file)
    
    user_prompt_file = 'plain_english_user_prompt.txt'
    user_prompt = reading_prompt(user_prompt_file)
    user_prompt = user_prompt.replace('{query_text}', query_text)
    user_prompt = user_prompt.replace('{content}', LLM_response)
    
    client = AzureOpenAI(api_key=os.getenv("openai_api_key"),
                         api_version=os.getenv("openai_api_version"),
                         azure_endpoint=os.getenv("openai_azure_endpoint"))    
    
    message_text = [{'role':'system', 'content':systems_prompts},
                    {'role':'user', 'content': user_prompt}]
    
    response = client.chat.completions.create(model='gpt-4o-code',
                                              messages=message_text).choices[0].message.content

    # print(response)
    return {"LLM_2_response": response}

def bulletpoint_formating(state: State):
    LLM_response = state['LLM_2_response']
    query_text = state['query']

    system_prompt_file = 'bulletpoints_system_prompt.txt'
    systems_prompts = reading_prompt(system_prompt_file)
    
    user_prompt_file = 'bulletpoints_user_prompt.txt'
    user_prompt = reading_prompt(user_prompt_file)
    user_prompt = user_prompt.replace('{query_text}', query_text)
    user_prompt = user_prompt.replace('{content}', LLM_response)

    client = AzureOpenAI(api_key=os.getenv("openai_api_key"),
                         api_version=os.getenv("openai_api_version"),
                         azure_endpoint=os.getenv("openai_azure_endpoint"))    
    
    message_text = [{'role':'system', 'content':systems_prompts},
                    {'role':'user', 'content': user_prompt}]
    
    response = client.chat.completions.create(model='gpt-4o-code',
                                              messages=message_text).choices[0].message.content

    # print(response)
    return {"output": response,
            "framework_numbers": re.findall(r'RM\d+', response)}

workflow = StateGraph(State)

# add all nodes.
workflow.add_node('query_supervisor', llm_call_router_stage_1)
workflow.add_node('Framework_Recommendation_agent', Framework_Recommendation_agent)  
workflow.add_node('Startup_agenet', Startup_agenet)
workflow.add_node('Comparison_agent', Comparison_agent)
workflow.add_node('query_out_of_scope', query_out_of_scope)
workflow.add_node('Framework_Supervisor', llm_call_router_stage_2)
workflow.add_node('Framework_details_agent', Framework_details_agent)
workflow.add_node('Framework_requery', Framework_requery)
workflow.add_node('plaing_english_numbers', plaing_english_numbers)
workflow.add_node("bulletpoint_formating", bulletpoint_formating)

# add all edges.
workflow.add_edge("Framework_Recommendation_agent", "plaing_english_numbers")
workflow.add_edge("Startup_agenet", "plaing_english_numbers")
workflow.add_edge("Comparison_agent", "plaing_english_numbers")
workflow.add_edge("query_out_of_scope", END)
workflow.add_edge("Framework_details_agent", "plaing_english_numbers")
workflow.add_edge("Framework_requery", END)
workflow.add_edge("plaing_english_numbers", "bulletpoint_formating")
workflow.add_edge("bulletpoint_formating", END)

# conditional edges
# Add conditional routing 
workflow.add_conditional_edges("query_supervisor", route_decision_stage_1, {"Framework_Recommendation_agent":"Framework_Recommendation_agent",
                                                                                   "Startup_agenet": "Startup_agenet",
                                                                                   "Comparison_agent":"Comparison_agent",
                                                                                   "query_out_of_scope": "query_out_of_scope",
                                                                                   "Framework_Supervisor":"Framework_Supervisor"})

workflow.add_conditional_edges("Framework_Supervisor", route_decision_stage_2, {"Framework_details_agent":"Framework_details_agent",
                                                                                   "Framework_requery": "Framework_requery"})

# set the entry point 
workflow.set_entry_point("query_supervisor")


# compile the workflow
Framwrork_app = workflow.compile()
# display(Image(app.get_graph().draw_mermaid_png()))

# --------------------------- Main app -----------------------------------------

def MultiAgent_Answering(query):
    # state = Framwrork_app.invoke({"query": query})
    # query_classification = state['query_classification']
    # print(query_classification)
    # answer_html = state["output"]
    # answer = state["LLM_2_response"]
    # framework_numbers =state["framework_numbers"]

    state = Framwrork_app.invoke({"query": query})
    query_classification = state.get('query_classification', 'unknown')
    print(query_classification)
    
    # Use LLM_2_response if available, otherwise fall back to output
    answer = state.get("LLM_2_response", state.get("output", "No response generated"))
    answer_html = state.get("output", "No HTML response generated")
    framework_numbers = state.get("framework_numbers", [])
    
    log_query_to_blob_v2(query, answer, query_classification)
    return answer, answer_html, framework_numbers



# QUERY = "how can i possibly buy some cleaning services"
# answer, answer_html, framework_numbers = MultiAgent_Answering(QUERY)
