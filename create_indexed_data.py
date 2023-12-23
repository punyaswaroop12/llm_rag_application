# Import necessary classes and functions from the llama_index and langchain libraries
from llama_index import (
    GPTVectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext,
    StorageContext,
    LLMPredictor,
    load_index_from_storage,
)
from langchain.chat_models import ChatOpenAI

# Import the openai library and os module to set the API key
import openai
import os

openai.api_key = os.environ['OPENAI_API_KEY']

# Notify the user that the document loading process has begun
print("started the loading document process...")

# Read the data from the specified directory. Change './boiler_docs/' to your desired path.
documents = SimpleDirectoryReader('data/aws-case-studies-and-blogs/aws-case-studies-and-blogs/').load_data()

# Initialize the LLMPredictor with the desired GPT-3.5-turbo model and temperature setting
llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"))

# Create a ServiceContext using the initialized predictor
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

# Notify the user that the indexing process has begun
print("started the indexing process...")

# Create an index using the loaded documents and the created service context
index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)


