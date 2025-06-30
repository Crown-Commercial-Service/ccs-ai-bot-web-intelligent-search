from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from openai import AzureOpenAI
import os
import pandas as pd
from datetime import datetime
import re
import numpy as np

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

# Azure Cognitive Search credentials
search_service_endpoint = os.getenv('azure_search_service_new_endpoint')
search_api_key = os.getenv('azure_search_new_api_key')
index_name = os.getenv('azure_index_wholedata')

search_client = SearchClient(endpoint=search_service_endpoint, index_name=index_name, credential=AzureKeyCredential(search_api_key))

# read the data. 
df_original =pd.read_csv(r'C:\Users\Naresh.Sampara\PycharmProjects\P16_Webpilot_API\Data\20250625_Whole_text_data_semantic_chunk.csv')
df = df_original.replace(np.nan, 'None')
count = 8900
for i in range(8900,len(df),100):
    print(i)
    df_slice = df.loc[i:i+100,:]

    # Prepare the data to upload
    actions = []
    for index, row in df_slice.iterrows():
        
        chunk_embedding = get_text_embeddings(row['text'])
        current_date = datetime.now().strftime("%Y%m%d")

        FileName = os.path.splitext(os.path.basename(row["doc_name"]))[0]

        FileName = os.path.splitext(os.path.basename(row["doc_name"]))[0]
        # First replace special characters with safe alternatives
        FileName = FileName.replace('/', '-')  # Replace / with -
        FileName = FileName.replace(' ', '_')  # Replace spaces with _
        # Then remove any remaining problematic characters (keeping letters, numbers, hyphens, underscores)
        FileName = re.sub(r'[^\w\-]', '', FileName)
        # Ensure the filename doesn't start with underscore
        if FileName.startswith('_'):
            FileName = 'doc' + FileName  # or just FileName.lstrip('_')

        # Remove leading underscores from the filename
        # FileName = re.sub(r'^_+', '', FileName)  # This removes one or more leading underscores
        ID = f"{FileName}--{current_date}-{count}"
        ID = re.sub(r'[^\w\-]', '_', ID)
        print(ID)
        print(count)
        count+=1


        action ={
            "id": ID,
            "content": row['text'],
            "Type": FileName, 
            "Framework": row['Framework'],
            "Filename": FileName, 
            "Framework_link": row["doc_link"],
            "Classification": row["classification"],
            "embeddings": chunk_embedding
        }

    #     actions.append(action)

    #     # upload the documents 
    # results = search_client.upload_documents(documents = actions)
    
    # print(f"Upload succeeded: {results[0].succeeded}, {i}")

