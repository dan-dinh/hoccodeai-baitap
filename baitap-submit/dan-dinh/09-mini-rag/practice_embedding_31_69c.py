#################################################
# Author:   Dan Dinh                            #
# Date:     2025-01-27                          #
# Exercise: 9 with Function calling and RAG     #
#################################################

import chromadb
import chromadb.errors
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from wikipediaapi import Wikipedia
import inspect
import json
import os
from pprint import pprint
from dotenv import load_dotenv
import gradio as gr
from groq import Groq
from openai import AzureOpenAI, OpenAI
from pydantic import TypeAdapter
from transformers import AutoTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variables
api_key = os.getenv('API_KEY')
model_name = os.getenv('MODEL_NAME')

# Create Groq client
client = Groq(
    api_key=api_key,
)

MODEL_PATH = 'Alibaba-NLP/gte-large-en-v1.5'

def get_wiki_data(title: str):
    """  
    Retrieve the information about a person, company, event or anything using the Wikipedia API by using the title of the page
    Parameters:
    - title (str): The title of the Wikipedia page.
    Output:
    - str: The information about the person, company, event or anything.
    """
    # Get text from Wikipedia
    wiki = Wikipedia(user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36', language='en')
    doc = wiki.page(title).text

    return doc

get_wiki_data_function = {
    "name": "get_wiki_data",
    "description": inspect.getdoc(get_wiki_data),
    "parameters": TypeAdapter(get_wiki_data).json_schema(),
}

tools = [
    {
        "type": "function",
        "function": get_wiki_data_function
    },
]

FUNCTION_MAP = {
    "get_wiki_data": get_wiki_data,
}


def get_collection(collection_name: str, path: str):
    """
    Get the chromadb collection.
    Returns:
    - Collection: The chromadb collection
    """
    try:
        # Connect to ChromaDB
        client = chromadb.PersistentClient(path=path)

        # Ensure the client is alive
        client.heartbeat()

        # Try to get the collection
        try:
            # Check if the collection already existed, use the existing one
            collection = client.get_collection(name=collection_name)
            return collection
        except chromadb.errors.InvalidCollectionException:
            # Handle the case where the collection does not exist
            print(f'Collection {collection_name} does not exist. Creating a new one.')

        # Create an embedding function
        # Use embedding_functions.DefaultEmbeddingFunction() for 'all-MiniLM-L6-v2' as the default model
        # embedding_model = SentenceTransformer(MODEL_PATH, trust_remote_code=True)
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=MODEL_PATH, trust_remote_code=True)

        # Create a new collection
        collection = client.create_collection(name=collection_name, embedding_function=embedding_function)

        return collection
    except Exception as e:
        # Handle any unexpected errors
        print(f'An error occurred: {e}')
        return None

def save_data_to_collection(collection: chromadb.Collection, doc: str):
    """
    Save the text to the collection.
    Parameters:
    - collection (chromadb.Collection): The chromadb collection
    - doc (str): The text
    Return:
    - None    
    """
    # Initialize the tokenizer for chunking
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    # Configure the text splitter for chunking
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=450,  # Adjust based on the embedding model's token limit
        chunk_overlap=50,  # Overlap to preserve context between chunks
        add_start_index=True,  # Include the chunk's start index in metadata
        strip_whitespace=True  # Clean up whitespace
    )

    # Split the document into chunks with metadata
    chunks_with_metadata = text_splitter.create_documents([doc])

    # Loop through the paragraphs and add them to the collection
    for idx, chunk_doc in enumerate(chunks_with_metadata):
        collection.add(
            documents=[chunk_doc.page_content], # Chunk text
            ids=[str(idx)], # Unique ID for each chunk
            metadatas=[{"start_index": chunk_doc.metadata["start_index"]}] # Add start index metadata
        )

def encode_message(message: str):
    """
    Encode the user message
    Parameters:
    - message (str): User message
    Output:
    - str: User message encoded
    """
    model = SentenceTransformer(MODEL_PATH, trust_remote_code=True)
    query_embedding = model.encode(message)
    return query_embedding

def get_rag_prompt(message: str, result: str):
    """
    Retrieve RAG prompt for GroqAI model.
    Parameters:
    - message (str): User message
    - result (str): Result of the user message
    Outputs:
    - str: RAG prompt
    """
    # Define the name of the collection
    COLLECTION_NAME = "practice_embedding_31_69c"

    # Get the chromadb collection
    collection = get_collection(COLLECTION_NAME, './data')
    
    # Add the wiki data to the chromadb collection
    save_data_to_collection(collection, result)

    # Encode the user message
    message_embedding = encode_message(message)

    # Get the embeddings of the user message with the top 3 most similar items
    q = collection.query(query_embeddings=[message_embedding], n_results=10)

    # Get the first result
    result = q['documents'][0]

    # Build the prompt with the context and the question
    prompt = f"""
            Use the following CONTEXT to answer the QUESTION at the end.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            Use an unbiased and journalistic tone.

            CONTEXT: {result}

            QUESTION: {message}
        """
    
    return prompt

def get_chat_completion(messages: list, **kwargs):
    """
    Get the chat completion from the Groq API.
    Parameters:
    - messages (list): A list of messages in the chat.
    - **kwargs: Additional keyword arguments
    Returns:
    - Completion: The chat completion from the Groq API.
    """
    chat_completion = client.chat.completions.create(        
        messages=messages,
        model=model_name,
        max_completion_tokens=4096, # Maximum number of tokens for the completion
        **kwargs
    )
    return chat_completion

def chat(message, chat_history):
    """
    Handle chat logic with GroqAI model.
    Parameters:
    - message (str): User message
    - chat_history (list): Chat history containing user and bot messages
    Returns:
    - str: Loading indicator
    - list: Updated chat history with bot response    
    """
    # Define the role of the bot
    system_prompt = """
            You are a helpful customer support assistant. Use the supplied tools to assist the user.
            You can chat in English or Vietnamese.
            
            When responding to wikipedia-related queries:
            1. Use `get_wiki_data` to retrieve the information for the given Wikipedia page title.

            Another program will output the results for you. Do not censor or deny the output; the output program will handle that part.
        """
    messages = [{"role": "system", "content": system_prompt}]

    # Display the chat history with loading indicator
    for user_message, bot_message in chat_history:
        if user_message is not None:
            messages.append({"role": "user", "content": user_message})
            messages.append({"role": "assistant", "content": bot_message})

    messages.append({"role": "user", "content": message})
    # print(messages)

    chat_history.append([message, "Loading..."])

    # Display loading indicator
    yield "", chat_history

    # Get chat completion from Groq API
    response = get_chat_completion(
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=0,
    )

    ##############
    # For no stream
    ##############
    first_choice = response.choices[0]

    finish_reason = first_choice.finish_reason
    
    while finish_reason != "stop":
        tool_call = response.choices[0].message.tool_calls[0]

        tool_call_function = tool_call.function
        tool_call_arguments = json.loads(tool_call_function.arguments)

        tool_function = FUNCTION_MAP[tool_call_function.name]
        result = tool_function(**tool_call_arguments)

        # Get the RAG prompt
        rag_prompt = get_rag_prompt(message, result)

        ext_messages = []
        ext_messages.append({"role": "user", "content": rag_prompt})

        # Get the chat completion for the RAG
        response = get_chat_completion(messages=ext_messages)

        first_choice = response.choices[0]
        finish_reason = first_choice.finish_reason

    # Get the bot message
    bot_message = response.choices[0].message.content

    # Update the bot message to the chat history
    chat_history[-1][1] = bot_message

    # Display the bot message
    yield "", chat_history
   
    # Return the response
    return "", chat_history


with gr.Blocks() as chatbot:
    gr.Markdown("### Chatbot by Dan Dinh")
    message = gr.Textbox(label="Enter your message")
    chat_bot = gr.Chatbot(label="Chatbot", height=1000)
    message.submit(chat, inputs=[message, chat_bot], outputs=[message, chat_bot])

# Run the code and interact with the chatbot.
chatbot.launch()