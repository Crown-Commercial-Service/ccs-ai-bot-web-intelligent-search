import pandas as pd
import numpy as np
from M1_FilestoSemanticChunking import fileToSemanticChunks
from utils.UsefulFunctions import get_url_link, process_odt_file, filename_to_title
import os

Frameworks = [
    {'folder_name': "RM1043.8", 'classification': 'details'},
    {'folder_name': "RM6098", 'classification': 'details'},
    {'folder_name': "RM6116", 'classification': 'details'},
    {'folder_name': "RM6187", 'classification': 'details'},
    {'folder_name': "RM6264", 'classification': 'details'},
    {'folder_name': "Startup", 'classification': 'startup'},
    {'folder_name': "Framework", 'classification': 'recommendation'}
]

# parent_folder = r'C:\Users\Naresh.Sampara\PycharmProjects\P16_Webpilot_API\Data'
parent_folder = r'C:\Users\Naresh.Sampara\PycharmProjects\P16_Webpilot_API\Data'

# Load the CSV
df_framework = pd.read_csv(r'C:\Users\Naresh.Sampara\PycharmProjects\P16_Webpilot_API\Data\FrameworkList.csv')

# Process each framework
dataframes = []
for i in Frameworks:
    print(i)
    if i['classification'] == 'details':
        folder_name = i['folder_name']
        framework_number = folder_name  # Same as folder name
        classification = i['classification']

        # Safely extract the framework name
        match = df_framework.loc[df_framework['Framework_Number'] == framework_number, 'Framework_Name']
        framework_name = match.iloc[0] if not match.empty else "Unknown"

        # print("Folder Name:", folder_name)
        # print("Framework Name:", framework_name)
        # print("Framework Number:", framework_number)
        # print("Classification:", classification)
        # print("-" * 40)

        dir_list = os.listdir(os.path.join(parent_folder, folder_name))
        
        for fileName in dir_list:
            filefullname = os.path.join(os.path.join(parent_folder, folder_name), fileName)

            clean_text = process_odt_file(filefullname)

            doc_link = get_url_link(clean_text)
            FileTitle = filename_to_title(fileName)

            # Process file
            df = fileToSemanticChunks(
                filename=filefullname,
                framework_Number=framework_number,
                framework_Name=framework_name,
                filetitle=FileTitle,
                doc_link=doc_link,
                classification=classification
            )

            if df is not None and not df.empty:
                dataframes.append(df)


    elif i['classification'] == 'startup':
        folder_name = i['folder_name']
        framework_number = folder_name  # Same as folder name
        classification = i['classification']

        # Safely extract the framework name
        match = df_framework.loc[df_framework['Framework_Number'] == framework_number, 'Framework_Name']
        framework_name = match.iloc[0] if not match.empty else "Unknown"

        # print("Folder Name:", folder_name)
        # print("Framework Name:", framework_name)
        # print("Framework Number:", framework_number)
        # print("Classification:", classification)
        # print("-" * 40)

        dir_list = os.listdir(os.path.join(parent_folder, folder_name))
        
        for fileName in dir_list:
            filefullname = os.path.join(os.path.join(parent_folder, folder_name), fileName)

            clean_text = process_odt_file(filefullname)

            doc_link = get_url_link(clean_text)
            FileTitle = filename_to_title(fileName)

            # Process file
            df = fileToSemanticChunks(
                filename=filefullname,
                framework_Number=framework_number,
                framework_Name=framework_name,
                filetitle=FileTitle,
                doc_link=doc_link,
                classification=classification
            )

            if df is not None and not df.empty:
                dataframes.append(df)

    elif i['classification'] == 'recommendation':
        print(i['classification'])
        folder_name = i['folder_name']
        
        classification = i['classification']

        child_folder = os.path.join(parent_folder, folder_name)

        for root, dirs, files in os.walk(child_folder):
            for file in files:
                source_file_path = os.path.join(root, file)

                framework_number = file[:-4]  # Same as folder name
                print(framework_number)
                # Safely extract the framework name
                match = df_framework.loc[df_framework['Framework_Number'] == framework_number, 'Framework_Name']
                framework_name = match.iloc[0] if not match.empty else "Unknown"
                doc_link = f"https://www.crowncommercial.gov.uk/agreements/{framework_number}"

                with open(source_file_path, 'r') as f:
                    text = f.read()

                df = pd.DataFrame({'text': [text],
                                   'summary': [text],
                                    'classification': [classification],
                                    'Framework': [framework_number],
                                    'framework_Name': [framework_name],
                                    'doc_name': [framework_name],
                                    'doc_link': [doc_link],
                                    'fileTitle': [framework_name]})

                if df is not None and not df.empty:
                    dataframes.append(df)
                
whole_chunks = pd.concat(dataframes, ignore_index=True)
output_file = '20250625_Whole_text_data_semantic_chunk.csv'
whole_chunks.to_csv(output_file, index=False)

