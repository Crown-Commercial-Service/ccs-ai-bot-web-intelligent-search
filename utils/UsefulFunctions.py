import os 
from datetime import datetime
import re
from azure.storage.blob import BlobServiceClient
import json

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


