from odf import text, teletype
from odf.opendocument import load
from odf.table import Table, TableRow, TableCell
import re

def read_odt_with_headings(file_path):
    """
    Function to read an ODT file and split content based on headings.
    Returns a list of tuples where each tuple contains a heading and its content.
    """
    doc = load(file_path)
    chunks = []
    
    current_chunk = []
    current_heading = None
    
    # Iterate through all paragraphs
    for paragraph in doc.getElementsByType(text.P):
        style_name = paragraph.getAttribute("stylename") or ""
        content = teletype.extractText(paragraph)
        
        # Check if this paragraph is a heading
        if "Heading" in style_name or "Header" in style_name:
            # If we have content for the previous heading, save it
            if current_heading is not None and current_chunk:
                chunks.append((current_heading, '\n'.join(current_chunk)))
                current_chunk = []
            
            current_heading = content
        elif content.strip():
            current_chunk.append(content)
    
    # Add the final chunk
    if current_heading is not None and current_chunk:
        chunks.append((current_heading, '\n'.join(current_chunk)))
    
    # If no headings were found, try to identify them heuristically
    if not chunks:
        all_paragraphs = [teletype.extractText(p) for p in doc.getElementsByType(text.P) if teletype.extractText(p).strip()]
        
        current_chunk = []
        current_heading = "Introduction"  # Default heading
        
        for para in all_paragraphs:
            # Heuristic: short lines (less than 60 chars) might be headings
            if len(para) < 60 and not para.endswith(('.', ',', ';', ':', '?')):
                if current_chunk:
                    chunks.append((current_heading, '\n'.join(current_chunk)))
                    current_chunk = []
                current_heading = para
            else:
                current_chunk.append(para)
        
        # Add the final chunk
        if current_chunk:
            chunks.append((current_heading, '\n'.join(current_chunk)))
    
    # If still no chunks, use all content as one chunk
    if not chunks:
        all_text = '\n'.join([teletype.extractText(p) for p in doc.getElementsByType(text.P) if teletype.extractText(p).strip()])
        chunks.append(("Document Content", all_text))
    
    return chunks

def read_odt_as_single_chunk(file_path):
    """
    Function to read an ODT file and extract all text content into a single chunk.
    Returns a string containing all the text from the document.
    """
    doc = load(file_path)
    
    # Extract all paragraphs
    all_paragraphs = []
    
    # Iterate through all paragraphs
    for paragraph in doc.getElementsByType(text.P):
        content = teletype.extractText(paragraph)
        if content.strip():  # Only include non-empty paragraphs
            all_paragraphs.append(content)
    
    # Join all paragraphs with newlines to create a single text chunk
    all_text = '\n'.join(all_paragraphs)
    
    # If no text was extracted, return a message
    if not all_text.strip():
        return "No text content found in the document."
    
    return all_text


def extract_tables_from_odt(file_path):
    """
    Function to extract tables from an ODT file.
    """
    doc = load(file_path)
    tables_data = []
    
    # Get all tables in the document
    for table in doc.getElementsByType(Table):
        table_data = []
        
        # Process each row
        for row in table.getElementsByType(TableRow):
            row_data = []
            
            # Process each cell in the row
            for cell in row.getElementsByType(TableCell):
                # Extract text from all paragraphs in the cell
                cell_text = ""
                for paragraph in cell.getElementsByType(text.P):
                    cell_text += teletype.extractText(paragraph) + " "
                
                row_data.append(cell_text.strip())
            
            if row_data:  # Only add non-empty rows
                table_data.append(row_data)
        
        if table_data:  # Only add non-empty tables
            tables_data.append(table_data)
    
    # If no tables found using the Table element, try to detect tabular data in text
    if not tables_data:
        # Get all text
        all_text = '\n'.join([teletype.extractText(p) for p in doc.getElementsByType(text.P) if teletype.extractText(p).strip()])
        
        # Simple pattern to detect tabular data (lines with multiple spaces or pipe characters)
        table_pattern = re.compile(r'((?:[^\n]+[\s\|]{3,}[^\n]+\n){3,})')
        table_matches = table_pattern.finditer(all_text)
        
        for match in table_matches:
            # Split into rows and clean up
            rows = match.group(0).strip().split('\n')
            table = [row.split('|') if '|' in row else re.split(r'\s{3,}', row) for row in rows]
            tables_data.append(table)
    
    return tables_data

def table_to_text(tables):
    """
    Convert table data to text format.
    Works for both ODT and PDF table extraction.
    """
    text_data = []
    
    for table in tables:
        # Handle both formats: list of rows (ODT) or string (PDF)
        if isinstance(table, str):
            # Table is already text (from PDF)
            text_data.append(table)
        else:
            # Table is a list of rows (from ODT)
            table_text = ""
            for row in table:
                row_text = " | ".join(row)
                table_text += row_text + "\n"
            text_data.append(table_text)
    
    return text_data