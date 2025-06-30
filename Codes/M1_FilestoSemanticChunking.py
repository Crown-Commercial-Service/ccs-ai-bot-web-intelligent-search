"""
This script processes various document formats (PDF, DOCX, ODT, TXT) by:
1. Classifying the file type
2. Extracting text content
3. Summarizing the document
4. Splitting text into sentences
5. Creating semantic chunks based on similarity
6. Storing results in a structured DataFrame

The workflow is implemented using LangGraph for state management and processing.
"""
import pandas as pd
import os
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from IPython.display import Image, display, Markdown
from utils.odt_file_reading import read_odt_as_single_chunk, extract_tables_from_odt, table_to_text
from utils.gpt_LLM import summarize_table_with_gpt
from openai import AzureOpenAI
from nltk.tokenize import sent_tokenize
import nltk
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
from docx import Document
import re

# Define the state structure for the workflow
class State(TypedDict):
    """State dictionary that maintains all processing data throughout the workflow"""
    fileName: str                      # Path to input file
    file_classification: str           # File type (PDF, DOCX, ODT, TXT)
    content_text: str                  # Extracted text content
    framework_Number: str              # Framework number metadata
    framework_Name: str                # Framework name metadata
    filetitle: str                     # Title of the document
    framework_classification: str       # Framework classification
    content: pd.DataFrame              # Final output DataFrame
    chunks: list                       # Semantic chunks of text
    textToSentenses: list              # Sentences extracted from text
    Summary: str                       # Document summary
    semantic_embeddings: list          # Embeddings for semantic analysis
    output: str                        # Processing status/output message
    doc_link: str                      # Document URL/link
    classification: str

def get_embeddings(text: str) -> list:
    """
    Generate embeddings for given text using Azure OpenAI
    
    Args:
        text: Input text to generate embeddings for
        
    Returns:
        List of embedding vectors
    """
    client = AzureOpenAI(
        api_key=os.getenv("openai_api_key"),
        api_version=os.getenv("openai_api_version"),
        azure_endpoint=os.getenv("openai_azure_endpoint")
    )
    
    embed = client.embeddings.create(
        model="text-embedding-ada-002", 
        input=text
    ).data[0].embedding
    
    return embed

def split_with_overlap(text, max_words=120, overlap=15):
    words = text.split()
    if len(words) <= max_words:
        return [text]
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+max_words]
        chunks.append(' '.join(chunk))
        i += max_words - overlap  # Move forward with overlap
        if i >= len(words):  # Avoid empty chunks at end
            break
    return chunks

def file_classification(state: State) -> dict:
    """
    Classify the input file based on its extension
    
    Args:
        state: Current workflow state
        
    Returns:
        Dictionary with file classification
    """
    filename = state['fileName'].lower()

    if filename.endswith('.pdf'):
        fileType = 'PDF'
    elif filename.endswith('.docx'):
        fileType = 'DOCX'
    elif filename.endswith('.odt'):
        fileType = 'ODT'
    elif filename.endswith('.txt'):
        fileType = 'TEXT'
    else:
        fileType = 'Out_of_scope'

    return {'file_classification': fileType}

def route_decsion_file_type_stage_1(state: State) -> str:
    """
    Route to appropriate processing node based on file type
    
    Args:
        state: Current workflow state
        
    Returns:
        Name of next processing node
    """
    file_classification = state.get("file_classification")

    if file_classification == "PDF":
        return "Read_PDF_File"
    elif file_classification == "DOCX":
        return "Read_Docx_File"
    elif file_classification == "ODT":
        return "Read_Odt_File"
    elif file_classification == "TEXT":
        return "Read_txt_File"
    else: 
        return "File_out_of_scope"
    
def read_PDF(state: State) -> dict:
    """
    Extract text content from PDF file
    
    Args:
        state: Current workflow state
        
    Returns:
        Dictionary with extracted text content
    """
    filename = state['fileName']
    text = ""

    with open(filename, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() or ''
    
    # Clean up whitespace
    text = ' '.join(text.split())
    return {'content_text': text}

def read_DOCX(state: State) -> dict:
    """
    Extract text content from DOCX file
    
    Args:
        state: Current workflow state
        
    Returns:
        Dictionary with extracted text content
    """
    filename = state['fileName']
    doc = Document(filename)
    text = " ".join([para.text for para in doc.paragraphs])
    # Clean up whitespace
    text = ' '.join(text.split())
    return {'content_text': text}

def read_ODT(state: State) -> dict:
    """
    Extract and process text content from ODT file including tables
    
    Args:
        state: Current workflow state
        
    Returns:
        Dictionary with extracted and processed text content
    """
    filename = state['fileName']
    # Extract main text
    Text_data = read_odt_as_single_chunk(filename).replace("\n", "")
    # Extract and process tables
    Tables_data = extract_tables_from_odt(filename)
    Tables_text = table_to_text(Tables_data)

    # Summarize each table
    Table_chunks = ""
    for table in Tables_text:
        table_chunk = summarize_table_with_gpt(table).replace("\n", "")
        Table_chunks = Table_chunks + table_chunk + ' '
    
    # Combine text and processed tables
    Text_data = Text_data + ' ' + Table_chunks 
    return {'content_text': Text_data}

def read_TXT(state: State) -> dict:
    """
    Extract text content from TXT file
    
    Args:
        state: Current workflow state
        
    Returns:
        Dictionary with extracted text content
    """
    filename = state['fileName']
    with open(filename, 'r') as txt_content:
        txt_chunk = txt_content.read()
    return {'content_text': txt_chunk}

def Summarize_doc(state: State) -> dict:
    """
    Generate a summary of the document using GPT-4
    
    Args:
        state: Current workflow state
        
    Returns:
        Dictionary with document summary
    """
    fileTitle = state['filetitle']
    FrameworkNumber = state['framework_Number']
    FrameworkName = state['framework_Name']

    system_Prompts = """
    You are an expert document summarizer. These document consists of the CCS framework information. 
    Your task is to analyze the given document and provide a concise yet comprehensive summary that includes:
    - Global Theme – The overarching subject or main idea of the document.
    - Key Topics – The primary subjects or arguments discussed.
    
    Structure your response as follows:
    Global Theme: [Briefly state the main theme]
    Key Topics: [List the core topics covered]
    Summary: [A 5 to 8 sentence overview capturing the essence of the content and file title 
             and if possible add about the framework information if it exists.]
    """

    user_prompt = f"""
    Summarize the following content and title of the content by identifying:
    - The global theme of the content description and title.
    - The key topics discussed in detail.
    
    After extraction, provide a concise summary (3-5 sentences) synthesizing the above elements.
    
    Content: {state['content_text']}
    Title: can you remove the file type and replace - with spaces in the {fileTitle}
    Framework Number: {FrameworkNumber}
    Framework Name: {FrameworkName}"""

    client = AzureOpenAI(
        api_key=os.getenv("openai_api_key"),
        api_version=os.getenv("openai_api_version"),
        azure_endpoint=os.getenv("openai_azure_endpoint")
    )
    
    message_text = [
        {'role': 'system', 'content': system_Prompts},
        {'role': 'user', 'content': user_prompt}
    ]
    
    response = client.chat.completions.create(
        model='gpt-4o-code',
        messages=message_text
    ).choices[0].message.content
    
    return {'Summary': response}

def TextToSentenses(state: State) -> dict:
    """
    Split text content into sentences using NLTK
    
    Args:
        state: Current workflow state
        
    Returns:
        Dictionary with list of sentences
    """
    text = state['content_text']
    # textToSentense = sent_tokenize(text)
    
    # Basic cleanup
    text = re.sub(r'\s+', ' ', text)

    # Rule-based segmentation
    # Break after periods, question marks, exclamation marks — not followed by lowercase or digit
    text = re.sub(r'(?<=[.?!])\s+(?=[A-Z0-9])', '\n', text)

    # Optional: Split overly long "sentences" further
    sentences = text.split('\n')
    refined = []
    for s in sentences:
        if len(s.split()) > 100:  # Arbitrary threshold for long sentences
            # Try breaking on semicolons and colons
            refined.extend(re.split(r'(?<=[;:])\s+', s))
        else:
            refined.append(s)
    textToSentense = [s.strip() for s in refined if s.strip()]

    return {'textToSentenses': textToSentense}

def SemanticChunking(state: State) -> dict:
    """
    Create semantic chunks from sentences based on cosine similarity of embeddings
    
    Args:
        state: Current workflow state
        
    Returns:
        Dictionary with list of semantic chunks
    """
    sentences = state['textToSentenses']
    similarity_threshold = 0.75  # Threshold for combining sentences
    max_words = 100              # Maximum words per chunk

    chunks = []
    current_chunk = []
    current_word_count = 0
    chunk_id = 1
    
    for sentence in sentences:
        sentence_word_count = len(sentence.split())
        
        # If sentence itself is too big, handle it separately
        if sentence_word_count > max_words:
            if current_chunk:
                chunks.append({
                    'chunk_id': chunk_id, 
                    'text': ' '.join(current_chunk), 
                    'word_count': current_word_count
                })
                chunk_id += 1
                current_chunk = []
                current_word_count = 0
            
            # Add the big sentence as its own chunk
            chunks.append({
                'chunk_id': chunk_id, 
                'text': sentence, 
                'word_count': sentence_word_count
            })
            chunk_id += 1
            continue
            
        # Calculate similarity if we have a current chunk
        if current_chunk:
            similarity = cosine_similarity(
                [get_embeddings(' '.join(current_chunk))], 
                [get_embeddings(sentence)]
            )[0][0]
        else:
            similarity = 1.0  # Force add to empty chunk
            
        # Check if we should add to current chunk
        if (similarity >= similarity_threshold and 
            current_word_count + sentence_word_count <= max_words):
            current_chunk.append(sentence)
            current_word_count += sentence_word_count
        else:
            # Save current chunk if not empty
            if current_chunk:
                chunks.append({
                    'chunk_id': chunk_id, 
                    'text': ' '.join(current_chunk), 
                    'word_count': current_word_count
                })
                chunk_id += 1
            # Start new chunk with current sentence
            current_chunk = [sentence]
            current_word_count = sentence_word_count
    
    # Add the last chunk if not empty
    if current_chunk:
        chunks.append({
            'chunk_id': chunk_id, 
            'text': ' '.join(current_chunk), 
            'word_count': current_word_count
        })

    return {'chunks': chunks}

def intodf(state: State) -> dict:
    """
    Convert processing results into structured DataFrame
    
    Args:
        state: Current workflow state
        
    Returns:
        Dictionary with output message and DataFrame
    """
    # Extract metadata
    fileTitle = state['filetitle']
    classification = state['classification']
    Framework = state['framework_Number']
    framework_Name = state['framework_Name']
    doc_name = state['fileName']
    doc_link = state['doc_link']

    # Create DataFrame from chunks
    chunks = state.get('chunks', [])
    summary = state.get('Summary', '')
    df = pd.DataFrame(chunks) if chunks else pd.DataFrame()

    processed_rows = []

    for _, row in df.iterrows():
        text = str(row['text'])  # Replace 'text' with your actual column name if different
        word_count = len(text.split())
        if word_count < 4:
            continue
        elif word_count > 120:
            for chunk in split_with_overlap(text):
                processed_rows.append({'text': chunk, 'word_count': len(chunk.split())})
        else:
            processed_rows.append({'text': text, 'word_count': len(text.split())})

    # Create new DataFrame
    df = pd.DataFrame(processed_rows)
    
    # Add metadata columns
    df['summary'] = summary
    df['classification'] =classification
    df['Framework'] = Framework
    df['framework_Name']= framework_Name
    df['doc_name'] = doc_name
    df['doc_link'] =  doc_link
    df['fileTitle'] = fileTitle

    response = f"""The {fileTitle} file contents is converted into chunks using semantic chunking and saved in a csv file."""
    return {'output': response, 'content': df}

def file_out_of_scope(state: State) -> dict:
    """
    Handle unsupported file types
    
    Args:
        state: Current workflow state
        
    Returns:
        Dictionary with error message
    """
    filename = state['fileName']
    response_message = f"""
    The file '{filename}' is not supported for index creation in Azure.
    Please upload files in one of the following formats: PDF, TXT, ODT, or DOCX.
    """
    return {"output": response_message, "content":pd.DataFrame()}
    

# Build the workflow using LangGraph
workflow = StateGraph(State)

# Add processing nodes
workflow.add_node('File_classifier', file_classification)
workflow.add_node('Read_PDF_File', read_PDF)
workflow.add_node('Read_Docx_File', read_DOCX)
workflow.add_node('Read_Odt_File', read_ODT)
workflow.add_node('Read_txt_File', read_TXT)
workflow.add_node('File_out_of_scope', file_out_of_scope)
workflow.add_node('Text_To_Sentenses', TextToSentenses)
workflow.add_node('Summarize_doc', Summarize_doc)
workflow.add_node('Semantic_Chunking', SemanticChunking)
workflow.add_node('Data_to_df', intodf)

# Define workflow edges
workflow.add_edge('Read_PDF_File', 'Summarize_doc')
workflow.add_edge('Read_Docx_File', 'Summarize_doc')
workflow.add_edge('Read_Odt_File', 'Summarize_doc')
workflow.add_edge('Read_txt_File', 'Summarize_doc')
workflow.add_edge('File_out_of_scope', END)
workflow.add_edge('Read_PDF_File', 'Text_To_Sentenses')
workflow.add_edge('Read_Odt_File', 'Text_To_Sentenses')
workflow.add_edge('Read_txt_File', 'Text_To_Sentenses')
workflow.add_edge('Summarize_doc', 'Data_to_df')
workflow.add_edge('Text_To_Sentenses', 'Semantic_Chunking')
workflow.add_edge('Semantic_Chunking', 'Data_to_df')
workflow.add_edge('Data_to_df', END)

# Add conditional routing based on file type
workflow.add_conditional_edges(
    'File_classifier', 
    route_decsion_file_type_stage_1, 
    {
        'Read_PDF_File': 'Read_PDF_File',
        'Read_Docx_File': 'Read_Docx_File',
        'Read_Odt_File': 'Read_Odt_File',
        'Read_txt_File': 'Read_txt_File',
        'File_out_of_scope': 'File_out_of_scope'
    }
)

# Set entry point and compile workflow
workflow.set_entry_point("File_classifier")
app = workflow.compile()

# Example usage
filename = r'C:\Users\Naresh.Sampara\PycharmProjects\P13_Webpilot\Data\RM1043.8\RM1043.8_Call-Off-Schedule-1-Transparency-Reports-v1.0.odt'
App = app.invoke({
    "fileName": filename,
    'filetitle': "RM1043.8 Call Off Schedule 1 Transparency Reports v1.0",
    'doc_link': 'https://www.crowncommercial.gov.uk/agreements/RM1043.8',
    'classification': 'Framework buying data',
    'framework_Number': 'RM1043.8',
    'framework_Name': 'Digital Outcomes 6'
})

# print(App['output'])
# print(App['content'])
# df = App['content']
# print(df.head())
# df.to_csv('test.csv')

def fileToSemanticChunks(filename:str, framework_Number:str, framework_Name:str, filetitle:str, doc_link:str, classification:str):

    App = app.invoke({
    "fileName": filename,
    'filetitle': filetitle,
    'doc_link': doc_link,
    'classification': classification,
    'framework_Number': framework_Number,
    'framework_Name': framework_Name})

    return App['content']
