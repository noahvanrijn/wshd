import os
import json

from apify_client import ApifyClient
from dotenv import load_dotenv
from langchain.document_loaders import ApifyDatasetLoader
from langchain_community.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

# Load the environment variables
load_dotenv()

#-----------------------KEYS-----------------------
APIFY_API_TOKEN = os.getenv('APIFY_API_TOKEN')

# Client side and and url of the website
apify_client = ApifyClient(APIFY_API_TOKEN)
website_url = "https://www.wshd.nl/"

print(f'Extracting data from "{website_url}". Please wait...')

# Initialize the actor 
actor_run_info = apify_client.actor('apify/website-content-crawler').call(
    run_input={'startUrls': [{'url': website_url}]}
)

print('Saving data into the vector database. Please wait...')

# Initialize the document loader
loader = ApifyDatasetLoader(
    dataset_id=actor_run_info['defaultDatasetId'],
    dataset_mapping_function=lambda item: Document(
        page_content=item['text'] or '', metadata={'source': item['url']}
    ),
)

# Load the documents and chunk them
docs = loader.load()

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
# docs = text_splitter.split_documents(documents)


# -----------------------SAVE THE DOCUMENTS FOR STORAGE-----------------------
# Define the file path
json_file_path = 'wshd.json'

# Prepare the data for JSON serialization
docs_json = [{'page_content': doc.page_content, 'metadata': doc.metadata} for doc in docs]

# Write the data to a JSON file
with open(json_file_path, 'w', encoding='utf-8') as file:
    json.dump(docs_json, file, indent=4)


# -----------------------SAVE THE DOCUMENTS AS EMBEDDING-----------------------
# Initialize the embedding and vector store
embedding = OpenAIEmbeddings()

# Create the vector database
vectordb = Chroma.from_documents(
    documents=docs,
    embedding=embedding,
    persist_directory='vector_db',
)

# Save the vector database
vectordb.persist()
print('All done!')