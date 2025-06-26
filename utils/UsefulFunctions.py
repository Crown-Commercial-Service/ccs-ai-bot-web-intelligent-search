import os 
from datetime import datetime
import re
from azure.storage.blob import BlobServiceClient
import json
from odf.opendocument import load
from odf.text import P
from odf.element import Node
from openai import AzureOpenAI

def log_query_to_blob(query, answer):
    connect_str = os.getenv('blob_storgae_connection_string')
    container_name = os.getenv('container_name')  
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    
    now = datetime.now()
    query_name = now.strftime("log_query_%Y-%m-%d_%H-%M-%S.json")
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=query_name)
    blob_client.upload_blob(json.dumps({'query':query,
                                        'LLM_response':answer}), overwrite=True)
    print('The query is logged')

def format_llm_response(llm_response):
    """
    Formats the LLM response for HTML rendering:
    - Converts numbered lists into <ul><li> elements.
    - Converts **text** into <strong>text>.
    """
    lines = llm_response.split('\n')
    formatted_lines = []
    for line in lines:
        # Check if the line starts with a numbered bullet point
        if re.match(r"^\d+\.", line.strip()):
            # Wrap in <li> and replace **text** with <strong>text</strong>
            line = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", line.strip())
            formatted_lines.append(f"<li>{line[3:].strip()}</li>")  # Skip "1. " at the start
        else:
            # Process lines not part of the numbered list (e.g., general text)
            line = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", line.strip())
            formatted_lines.append(f"<p>{line}</p>")

    # Wrap the numbered items in <ul> if present
    if formatted_lines:
        formatted_content = "<ul>" + "".join(formatted_lines) + "</ul>"
    else:
        formatted_content = "<p>No content provided.</p>"

    return formatted_content


def reading_prompt(filename):
    blob_string = os.getenv("blob_storgae_connection_string")
    container_Name = 'webpilot-prompts'

    blob_server_connection = BlobServiceClient.from_connection_string(blob_string)

    blob_client = blob_server_connection.get_blob_client(container=container_Name, blob=filename)

    blob_data = blob_client.download_blob()
    prompt_text = blob_data.readall().decode('utf-8')
    # print(prompt_text)

    return prompt_text


def log_query_to_blob_v2(query, answer, query_classification):

    connect_str = os.getenv('blob_storgae_connection_string')
    container_name = os.getenv('container_name')  
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    
    now = datetime.now()
    query_name = now.strftime("log_query_%Y-%m-%d_%H-%M-%S.json")
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=query_name)
    blob_client.upload_blob(json.dumps({'query':query,
                                        'LLM_response':answer, 
                                        'query_classification':query_classification}), overwrite=True)
    

def get_url_link(text):
    
    client = AzureOpenAI(
        api_key=os.getenv("openai_api_key"),
        api_version=os.getenv("openai_api_version"),
        azure_endpoint=os.getenv("openai_azure_endpoint")
    )
    
    system_prompt = """You are an expert at extracting specific information from text."""
    user_prompt = f"""Please extract the URL from the following content and return it as the Source URL.
    Content: {text}
    Return only the URL."""

    message_text = [
        {'role':'system', 'content':system_prompt},
        {'role':'user', 'content': user_prompt}
    ]
    
    response = client.chat.completions.create(
        model='gpt-4o-code',
        messages=message_text,
        max_tokens=100
    ).choices[0].message.content
    
    return response.strip()

def process_odt_file(file_path):
    """Process ODT file and extract text safely"""
    try:
        textdoc = load(file_path)
        all_paragraphs = textdoc.getElementsByType(P)
        
        text_parts = []
        for paragraph in all_paragraphs:
            try:
                if isinstance(paragraph.firstChild, Node) and paragraph.firstChild.data:
                    text_parts.append(paragraph.firstChild.data)
            except AttributeError:
                continue
        
        raw_text = " ".join(text_parts)
        clean_text = ' '.join(raw_text.split())  # Clean text
        return clean_text
    except Exception as e:
        return ""


def filename_to_title(filename):
    """Converts filename to a clean title format"""
    try:
        # Remove the file extension
        name_without_ext = os.path.splitext(filename)[0]
        # Replace underscores with spaces
        name_with_spaces = name_without_ext.replace('_', ' ')
        # Capitalize the first letter of the string
        clean_title = name_with_spaces.capitalize()
        return clean_title
    except Exception as e:
        return filename  # Return original as fallback
