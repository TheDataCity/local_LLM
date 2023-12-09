import logging
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import pandas as pd

import click
import torch
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from constants import (
    CHROMA_SETTINGS,
    DOCUMENT_MAP,
    EMBEDDING_MODEL_NAME,
    INGEST_THREADS,
    PERSIST_DIRECTORY,
    SOURCE_DIRECTORY,
)

def create_embedding(device_type):
    # Create embeddings
    embedding = HuggingFaceInstructEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": device_type},
        )
    # change the embedding type here if you are running into issues.
    # These code uses much smaller embeddings and will work for most appications
    # If you use HuggingFaceEmbeddings, make sure to also use the same in the
    # run_localGPT.py file.

    # embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    return embedding

def load_document_path(source_dir):
    # Loads all documents from the source documents directory
    fileNameList = os.listdir(source_dir)
    # Get the full file paths
    filePaths = [os.path.join(source_dir, fileName) for fileName in fileNameList]  
    return filePaths

# Function to extract company info from a single text file
def extract_company_info(file_path):

    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    for line in lines:
        if line.startswith("CompanyName :: "):
            companyName = line.split("::")[1].strip()

        if line.startswith("company_number :: "):
            companyNumber = line.split("::")[1].strip()

    if companyName and companyNumber:
        return companyName, companyNumber
    else:
        return "Company information not found"


def create_option_menu(filePaths):
    optionList = []
    companyNameList = []
    companyNumberList = []
    for filePath in filePaths:
        companyName, companyNumber = extract_company_info(filePath)
        companyNameList.append(companyName)
        companyNumberList.append(companyNumber)
        optionList.append("%s (%s)" % (companyName.title(),companyNumber))

    optionList_dataFrame = pd.DataFrame({'Company Names with Company numbers': optionList})
    optionList_dataFrame.to_csv('data/Company_Names_with_Company_Numbers(Options).csv', index=False)
    print("Option menu created sucessfully")

    return companyNameList,companyNumberList

def create_db_from_text_files(filePaths,embedding):

    # Create option menu
    _, companyNumberList = create_option_menu(filePaths)
    counter = 0
    for companyNumber in companyNumberList:
        counter +=1
        if (not os.path.exists(os.path.join(PERSIST_DIRECTORY, companyNumber))):
            print(f"{counter}. {companyNumber} vector database doese not exisits")
            # Load entire Folder
            # text_loader_kwargs={'autodetect_encoding': True}
            # loader = DirectoryLoader("../database-sample/", glob="./*.txt",  show_progress=True, loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)

            # Load one company file
            try:
                file_name = f"company_{companyNumber}_info.txt"
                file_path = os.path.join(SOURCE_DIRECTORY, file_name)
                file_extension = os.path.splitext(file_path)[1]
                loader_class = DOCUMENT_MAP.get(file_extension)
                if loader_class:
                    print(file_path + ' loaded.')
                    loader = loader_class(file_path, encoding = 'UTF-8')
                else:
                    print(file_path + ' document type is undefined.')
                    raise ValueError("Document type is undefined")
            except Exception as ex:
                print('%s loading error: \n%s' % (file_path, ex))

            documents = loader.load()

            # Split the text into different chunks
            textSplitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            docs = textSplitter.split_documents(documents)

            # Save to database
            Chroma.from_documents(
                docs, 
                embedding, 
                persist_directory = os.path.join(PERSIST_DIRECTORY, companyNumber),
                client_settings=CHROMA_SETTINGS)
            print(f"{counter}. {companyNumber} vector database created")
        
        else:
            print(f"{counter}. {companyNumber} vector database already exisits")
        
    print("Vector database created")

@click.command()
@click.option(
    "--device_type",
    default="cuda" if torch.cuda.is_available() else "cpu",
    type=click.Choice(
        [
            "cpu",
            "cuda",
            "ipu",
            "xpu",
            "mkldnn",
            "opengl",
            "opencl",
            "ideep",
            "hip",
            "ve",
            "fpga",
            "ort",
            "xla",
            "lazy",
            "vulkan",
            "mps", #mac
            "meta",
            "hpu",
            "mtia",
        ],
    ),
    help="Device to run on. (Default is cuda)",
)
def main(device_type):
    # Load documents and split in chunks
    print(f"Loading documents from {SOURCE_DIRECTORY}")
    documentPathList = load_document_path(SOURCE_DIRECTORY)

    embeddingUsed = create_embedding(device_type)

    # Creating vector database
    create_db_from_text_files(documentPathList,embeddingUsed)

if __name__ == "__main__":
    main()
