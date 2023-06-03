'''
Here's a summary of what the script does:

Imports: The script starts by importing the necessary modules. os and streamlit are standard Python libraries, while the langchain modules are used for loading and processing documents.

API Key: It sets the environment variable OPENAI_API_KEY to a value fetched from Streamlit's secrets, which securely stores sensitive information.

Persist Directory: It sets a variable persist_directory as 'db', which is probably used to store processed data.

Document Loaders: It uses DirectoryLoader from langchain.document_loaders to load documents from specified directories.
It uses glob="*.pdf" to select all PDF files in the directory. It creates two document loaders: one for a directory named buffett and another for a directory named branson.

Load Documents: It loads documents using the loaders defined above into buffett_docs and branson_docs respectively.

Embeddings and Text Splitter: It creates an instance of OpenAIEmbeddings to generate embeddings for the documents.
It also creates an instance of CharacterTextSplitter which is probably used to split the text of the documents into chunks of 250 characters each with an overlap of 8 characters.

Split Documents and Generate Embeddings: It applies the text splitter on the documents to create smaller chunks of texts. The result is stored in buffett_docs_split and branson_docs_split respectively.

Create Chroma Instances and Persist Embeddings: It creates two instances of Chroma from langchain.vectorstores. Chroma is likely a kind of vector store used to store and retrieve embeddings for chunks of text. It uses the split documents and the OpenAIEmbeddings to initialize the Chroma instances. The embeddings for the chunks of text in the documents are generated and stored (persisted) in directories under the persist_directory using the persist method of the Chroma instances.
'''



import os
import streamlit as st
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Set persist directory
persist_directory = 'db'

buffett_loader = DirectoryLoader('./docs/buffett/', glob="*.pdf")
branson_loader = DirectoryLoader('./docs/branson/', glob="*.pdf")

buffett_docs = buffett_loader.load()
branson_docs = branson_loader.load()

embeddings = OpenAIEmbeddings()
text_splitter = CharacterTextSplitter(chunk_size=250, chunk_overlap=8)

# Split documents and generate embeddings
buffett_docs_split = text_splitter.split_documents(buffett_docs)
branson_docs_split = text_splitter.split_documents(branson_docs)

# Create Chroma instances and persist embeddings
buffettDB = Chroma.from_documents(buffett_docs_split, embeddings, persist_directory=os.path.join(persist_directory, 'buffett'))
buffettDB.persist()

bransonDB = Chroma.from_documents(branson_docs_split, embeddings, persist_directory=os.path.join(persist_directory, 'branson'))
bransonDB.persist()
