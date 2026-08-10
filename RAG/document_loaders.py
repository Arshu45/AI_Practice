# WHY ?
# Document Loaders are used to load and preprocess documents from various sources (like PDFs, text files, web pages, etc.) into a format that can be easily used for tasks like question answering, summarization, etc. They help in converting unstructured data into structured data that can be fed into language models.

import os
import tempfile
from pathlib import Path
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredHTMLLoader


from dotenv import load_dotenv

load_dotenv()


# Load text file
def load_text_file(file_path: str):
    loader = TextLoader(file_path)
    return loader.load()

# Load PDF file
def load_pdf_file(file_path: str):
    loader = PyPDFLoader(file_path)
    return loader.load()

# Print the content of the loaded documents

def print_documents(documents):
    print(f"Length: {len(documents)}")
    for i, doc in enumerate(documents):
        print(f"Document {i + 1}:")
        print(f"Metadata: {doc.metadata}")
        print(doc.page_content)
        print("-" * 40)

# print_documents(load_text_file("sample.txt"))
print_documents(load_pdf_file("docs/langchain_demo.pdf"))