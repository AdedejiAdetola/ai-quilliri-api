#!/usr/bin/env python
# coding: utf-8

import os


# install openai (version 0.27.1)
os.system('pip install openai==0.27.1')
# install langchain (version 0.0.191)
os.system('pip install langchain==0.0.191')
# install chromadb
os.system('pip install chromadb==0.3.26')
# install tiktoken
os.system('pip install tiktoken==0.4.0')



# Path to the text file containing links
#links_file = r'C:\Users\USER\OneDrive\Documents\ML\final_year\mayoclinic_links.txt'
# Output directory for downloaded files
output_directory = r'C:\Users\USER\OneDrive\Documents\ML\final_year\meddocs1'

# Read links from the file with 'utf-8' encoding
# with open(links_file, 'r', encoding='utf-8') as file:
#     links = file.readlines()


links_list = ['https://www.mayoclinic.org/', 'https://www.healthline.com/', 'https://openmd.com/', 'https://www.medicinenet.com/', 'https://medlineplus.gov/']
# Download each link
# for link in links_list:
#     link = link.strip()  # Remove leading/trailing whitespaces
#     print(f"Downloading: {link}")
    # Construct wget command
    # os.system('wget -r -P {output_directory} {link}')


import magic

def get_file_type(file_path):
    # Create a magic.Magic object
    detector = magic.Magic()

    # Get the file type
    file_type = detector.from_file(file_path)

    return file_type


import magic
import os

# def get_file_type(file_path):
#     # Create a magic.Magic object
#     detector = magic.Magic()

#     # Get the file type
#     file_type = detector.from_file(file_path)

#     return file_type

# def rename_to_html(file_path):
#     # Check if the file type contains "HTML document"
#     if "HTML document" in get_file_type(file_path):
#         # Append .html to the file
#         new_file_path = file_path + ".html"
#         os.rename(file_path, new_file_path)
#         print(f"Renamed '{file_path}' to '{new_file_path}'")

# # Example usage:
# # file_path = r'C:\Users\USER\OneDrive\Documents\ML\final_year\meddocs\www.mayoclinic.org\ar.1'
# # rename_to_html(file_path)


# # In[ ]:


# import magic
# import os

# def get_file_type(file_path):
#     # Create a magic.Magic object
#     detector = magic.Magic()

#     # Get the file type
#     file_type = detector.from_file(file_path)

#     return file_type

# def rename_html_files_in_directory(directory):
#     # Recursively iterate through all files and folders in the directory
#     for root, dirs, files in os.walk(directory):
#         for file_name in files:
#             file_path = os.path.join(root, file_name)
#             # Check if the file type contains "HTML document"
#             if "HTML document" in get_file_type(file_path):
#                 # Append .html to the file
#                 new_file_path = file_path + ".html"
#                 os.rename(file_path, new_file_path)
#                 print(f"Renamed '{file_path}' to '{new_file_path}'")


import os
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import BSHTMLLoader

# Path to the main directory containing subdirectories
main_path = r'C:\Users\grey\Documents\Final Year Project\final_year\med_documnetss'

# List all subdirectories in the main directory
subdirs = [os.path.join(main_path, d) for d in os.listdir(main_path) if os.path.isdir(os.path.join(main_path, d))]

# Initialize a list to store documents from all subdirectories
all_docs = []

# Loop through each subdirectory and load documents
for subdir in subdirs:
    loader = DirectoryLoader(subdir, glob="**/*.html", loader_cls=BSHTMLLoader, silent_errors=True, show_progress=True)
    docs = loader.load()
    all_docs.extend(docs)  # Append loaded documents to the list

# Now `all_docs` contains documents from all subdirectories


len(all_docs)


# In[16]:


all_docs


# In[18]:


# Import RecursiveCharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Create the text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

# Split the documents
documents = text_splitter.split_documents(all_docs)


# In[12]:


# Import tiktoken
import tiktoken

# Create an encoder 
encoder = tiktoken.encoding_for_model('text-embedding-ada-002')

# Count tokens in each document
doc_tokens = [len(encoder.encode(d.page_content)) for d in documents]

# Calculate the sum of all token counts
total_tokens = sum(doc_tokens)

# Calculate a cost estimate
cost_estimate = (sum(doc_tokens)/1000) * 0.0004
print(f"{total_tokens} tokens, cost estimate: ${cost_estimate:.2f}")


# In[19]:


import os
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access the OpenAI API key
openai_api_key = os.getenv('OPENAI_API_KEY')

# Optionally, set it in the environment if needed
os.environ['OPENAI_API_KEY'] = openai_api_key

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
# from langchain_text_splitters import CharacterTextSplitter
from tqdm import tqdm

print("Imports completed successfully.")

# create the open-source embedding function
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
print("Embedding function created successfully.")

# Initialize an empty Chroma object
db = Chroma(embedding_function=embedding_function)

# Actively track the loading state
print("Loading documents into Chroma...")
for doc in tqdm(documents, desc="Loading documents"):
    db.from_documents([doc], embedding_function)
print("Documents loaded into Chroma successfully.")

# query it
query = "What did the president say about Ketanji Brown Jackson"
docs = db.similarity_search(query)
print("Query executed successfully.")

# print results
print(docs[0].page_content)

# Call the `similarity_search_with_score` method on `db`
results = db.similarity_search_with_score("tell me about alzheimer's?")

# Print the results
for (doc, score) in results:
    print(f"Score: {score}\n{doc.page_content}\n----")


# In[60]:


# Import 
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chat_models import ChatOpenAI
import pandas

# Set the question variable
#what mental health conditions are related to lack of appetite and uneasiness
# question = "what is anxiety"
#input("Enter your question: ")
question = 'what mental health conditions are related to lack of appetite and uneasiness'


# Query the database as store the results as `context_docs`
context_docs = db.similarity_search(question)

# Create a prompt with 2 variables: `context` and `question`
prompt = PromptTemplate(
    template=""""Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

<context>
{context}
</context>

Question: {question}
Helpful Answer:""",
    input_variables=["context", "question"]
)

# Create an LLM with ChatOpenAI
llm = ChatOpenAI(temperature=0)

# Create the chain
qa_chain = LLMChain(llm=llm, prompt=prompt)

# Call the chain
result = qa_chain({"context": "\n".join([d.page_content for d in context_docs]), "question": question})

# Print the result
print(result["text"])



# In[ ]:




