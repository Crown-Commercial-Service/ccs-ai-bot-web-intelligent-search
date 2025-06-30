from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
import os

# Replace these with your actual values
endpoint = os.getenv('azure_search_service_new_endpoint')
index_name = os.getenv('azure_index_wholedata')
api_key = os.getenv('azure_search_new_api_key')
document_id = "Technology_Products__Associated_Services_2--20250626-9073"  # e.g., "12345"

# Create a SearchClient
search_client = SearchClient(
    endpoint=endpoint,
    index_name=index_name,
    credential=AzureKeyCredential(api_key)
)

# Fetch document by ID
try:
    result = search_client.get_document(key=document_id)
    print("Document Retrieved:")
    print(f"ID: {result.get('id')}")
    print(f"Content: {result.get('content')}")
    print(f"Type: {result.get('Type')}")
    print(f"Framework: {result.get('Framework')}")
except Exception as e:
    print(f"Error retrieving document: {e}")
