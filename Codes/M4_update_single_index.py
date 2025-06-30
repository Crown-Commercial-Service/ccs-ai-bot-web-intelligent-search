from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
import os
from openai import AzureOpenAI
import pandas as pd

# use full functions
def get_text_embeddings(text):
    client = AzureOpenAI(api_key=os.getenv('openai_api_key'),
                     api_version=os.getenv('openai_api_version'),
                     azure_endpoint=os.getenv('openai_azure_endpoint'))
    if pd.isna(text):
        print('text is None')
        response = None
    else: 
        response = client.embeddings.create(model = "text-embedding-ada-002", input = text). data[0].embedding

    return response
# 
endpoint = os.getenv('azure_search_service_new_endpoint')
index_name = os.getenv('azure_index_wholedata')
api_key = os.getenv('azure_search_new_api_key')

"""
8354,Technology_Products__Associated_Services_2--20250626-8964,recommendation
9183,Technology_Products__Associated_Services_2--20250626-9073,recommendation
6091,Technology_Products__Associated_Services_2--20250626-9175,details
7591,Technology_Products__Associated_Services_2--20250626-9180,details

"""


document_id = "Technology_Products__Associated_Services_2--20250626-9073"  # e.g., "12345"

# Create the search client
search_client = SearchClient(
    endpoint=endpoint,
    index_name=index_name,
    credential=AzureKeyCredential(api_key)
)

# Define updated content and embeddings
filename = r'C:\Users\Naresh.Sampara\PycharmProjects\P16_Webpilot_API\Data\Framework\Benefits\RM6098.txt'
with open(filename, 'r') as f:
    text = f.read()

print(text)

Framework = "RM6098"
Framework_link = "https://www.crowncommercial.gov.uk/agreements/RM6098"
Classification = "recommendation"
Filename = "RM6098-benefits"
# Construct the updated document
updated_doc ={"id": document_id,
                "content": text,
                "Type": Filename,
                "Framework": Framework,
                "Filename": Filename, 
                "Framework_link": Framework_link,
                "Classification": Classification,
                "embeddings": get_text_embeddings(text)
        }



# Upload (update) the document
result = search_client.upload_documents(documents=[updated_doc])

# Print status
if result[0].succeeded:
    print(f"Document {document_id} updated successfully.")
else:
    print(f"Failed to update document: {result[0].error_message}")
