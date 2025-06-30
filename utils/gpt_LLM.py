from openai import AzureOpenAI
import os
# from langchain.text_splitter import RecursiveCharacterTextSplitter

def summarize_table_with_gpt(text):
    Client = AzureOpenAI(api_key=os.getenv('openai_api_key'),
                         api_version="2023-07-01-preview",
                         azure_endpoint=os.getenv('openai_azure_endpoint'))
    
    message_text = [{'role':'system', 'content':"Summarize the following text focusing on tables and key points."},
                    {'role':'user','content':text}]
    
    response = Client.chat.completions.create(
        model='gpt-4o-code',
        messages=message_text,
        temperature=0.8
    ).choices[0].message.content

    return response


# # Function to split text using LangChain
# def split_text_langchain(text, chunk_size=1050, overlap=150):
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size, chunk_overlap=overlap, separators=[" ", "\n"]
#     )
#     return splitter.split_text(text)

