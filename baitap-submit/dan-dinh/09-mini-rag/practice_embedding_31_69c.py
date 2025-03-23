#################################################
# Author:   Dan Dinh                            #
# Date:     2025-01-27                          #
# Exercise: 9 with Function calling and RAG     #
#################################################

import uuid
import chromadb
import chromadb.errors
from chromadb.utils import embedding_functions
from wikipediaapi import Wikipedia
import inspect
import json
import os
from dotenv import load_dotenv
import gradio as gr
from groq import Groq
from pydantic import TypeAdapter
from transformers import AutoTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ragatouille import RAGPretrainedModel

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variables
api_key = os.getenv('API_KEY')
model_name = os.getenv('MODEL_NAME')

# Create Groq client
client = Groq(
    api_key=api_key,
)

# Define the name of the collection
COLLECTION_NAME = "practice_embedding_31_69c"
MODEL_PATH = 'Alibaba-NLP/gte-large-en-v1.5'

# Global cache to store wiki data for already fetch titles
wiki_data_cache = {} # Added cache to avoid redundant API calls

# Global variable to store the collection instance
db_collection = None

def generate_document_id(content: str):
    """
    Generate a unique ID for each document based on its content using UUID5.
    This ensures that duplicate documents (with the same content) do not get inserted again.
    Parameters:
    - content (str): The content of the document.
    Output:
    - str: The unique document ID.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, content))

def get_wiki_data(title: str):
    """  
    Retrieve the information about a person, company, event or anything using the Wikipedia API by using the title of the page
    Parameters:
    - title (str): The title of the Wikipedia page.
    Output:
    - str: The information about the person, company, event or anything.
    """
    # Check if the data for the given title already exists in the cache
    if title in wiki_data_cache:
        return wiki_data_cache[title]

    # Get text from Wikipedia
    wiki = Wikipedia(user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36', language='en')
    doc = wiki.page(title).text

    # Store the fetched data in the cache
    wiki_data_cache[title] = doc

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
            embedding_model = embedding_functions.DefaultEmbeddingFunction()
            
            # Create a new collection
            collection = client.create_collection(name=collection_name, embedding_function=embedding_model)

            return collection
    except Exception as e:
        # Handle any unexpected errors
        print(f'An error occurred: {e}')
        return None

def save_data_to_collection(collection: chromadb.Collection, doc: str, title: str):
    """
    Save the text to the collection.
    Parameters:
    - collection (chromadb.Collection): The chromadb collection
    - doc (str): The text
    - title (str): The title of the Wikipedia page
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
        # Generate a unique ID for each chunk based on its content
        chunk_id = generate_document_id(chunk_doc.page_content)

        # Check if the document with the generated ID already exists in the collection
        existing_document = collection.get(ids=[chunk_id])

        # Check if the document is not found (i.e., `existing_document` would be empty)
        if not existing_document or len(existing_document["documents"]) == 0:
            collection.add(
                documents=[chunk_doc.page_content], # Chunk text
                ids=[chunk_id], # Use the generated unique ID for each chunk
                metadatas=[{"start_index": chunk_doc.metadata["start_index"], "title": title}] # Add start index metadata and store title in metadata
            )
        else:
            print(f"Document with ID {chunk_id} already exists in the collection.")

def get_rag_prompt(message: str):
    """
    Retrieve RAG prompt for GroqAI model.
    Parameters:
    - message (str): User message
    - result (str): Result of the user message
    Outputs:
    - str: RAG prompt
    """
    # Get the collection (no population here)
    collection = get_collection(COLLECTION_NAME, "./data")
    if collection is None:
        raise Exception("Failed to initialize ChromaDB collection") 

    # Get the embedding function
    embedding_function = collection._embedding_function
    
    # Encode the user message
    message_embedding = embedding_function([message])[0]

    # Get the embeddings of the user message with the top 10 most similar items
    result_data = collection.query(query_embeddings=[message_embedding], n_results=10)

    # Get the first document from the result data
    results = result_data['documents'][0]

    # Initialize the RAG model
    reranker = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

    # Apply the reranker to narrow down the search results
    result = reranker.rerank(query=message, documents=results, k=3)

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

def is_data_in_db(collection: chromadb.Collection, title: str):
    """
    Check if chunks for a specific Wikipedia page title exist in the collection.
    Parameters:
    - collection (chromadb.Collection): The chromadb collection
    - title (str): The title of the Wikipedia page
    Returns:
    - bool: True if data exists in the collection, False otherwise
    """
    # Query the collection using the title stored in metadata
    existing_docs = collection.get(where={"title": title})
    return len(existing_docs["documents"]) > 0

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
            2. Avoid calling `get_wiki_data` repeatedly for the same title to prevent unnecessary API calls.

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
    
    # Get the collection
    collection = get_collection(COLLECTION_NAME, "./data")

    while finish_reason != "stop":
        tool_call = first_choice.message.tool_calls[0]

        tool_call_function = tool_call.function
        tool_call_arguments = json.loads(tool_call_function.arguments)

        tool_function = FUNCTION_MAP[tool_call_function.name]
        result = tool_function(**tool_call_arguments)

        # Save fetched data to ChromaDB if not already present
        title = tool_call_arguments['title']
        if not is_data_in_db(collection, title):
            save_data_to_collection(collection, result, title)
            print(f"Saved new data for {title} to ChromaDB.")

        # Append tool call and result to messages for context
        messages.append(first_choice.message)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": tool_call_function.name,
            "content": json.dumps(result)
        })

        # Get the chat completion for the RAG
        response = get_chat_completion(messages=messages)

        first_choice = response.choices[0]
        finish_reason = first_choice.finish_reason

    # Get the RAG prompt
    rag_prompt = get_rag_prompt(message)

    # Set the RAG prompt
    ext_messages = [{"role": "user", "content": rag_prompt}]

    # Get the chat completion for the RAG
    response = get_chat_completion(messages=ext_messages)
    
    # Get the bot message
    bot_message = response.choices[0].message.content

    # Update the bot message to the chat history
    chat_history[-1][1] = bot_message

    # Display the bot message
    yield "", chat_history
   
    # Return the response
    return "", chat_history

def populate_database(collection_name: str, path: str, wiki_titles: list):
    """
    Pre-build the ChromaDB with Wikipedia data for the given titles.
    This runs offline before the chatbot starts.
    Parameters:
    - collection_name (str): The name of the collection to be created.
    - path (str): The path to the database file.
    - wiki_titles (list): A list of Wikipedia page titles to populate the database with.
    Returns:
    - None
    """
    collection = get_collection(collection_name, path)
    if collection is None:
        raise Exception('Failed to initialize ChromaDB collection!')

    for title in wiki_titles:
        # Check if chunks for this title already exist in the collection
        existing_chunks = collection.get(where={'title': title})["ids"]
        if not existing_chunks:
            print(f"Data for {title} not found. Fetching and saving...")
            doc = get_wiki_data(title)

            # Save to ChromaDB
            save_data_to_collection(collection, doc, title)
        else:
            print(f"Data for {title} already exists in the database. Skipping fetch and save.")

    print("Database population completed.")

if __name__ == "__main__":
    """
    Main function to run the program.
    """
    # Pre-populate the database with the specified Wikipedia pages
    wiki_titles = ["Sơn_Tùng_M-TP", "Jujutsu_Kaisen"]
    populate_database(COLLECTION_NAME, "./data", wiki_titles)

    # Launch the chatbot
    with gr.Blocks() as chatbot:
        gr.Markdown("### Chatbot by Dan Dinh")
        message = gr.Textbox(label="Enter your message")
        chat_bot = gr.Chatbot(label="Chatbot", height=1000)
        message.submit(chat, inputs=[message, chat_bot], outputs=[message, chat_bot])

    # Run the code and interact with the chatbot.
    chatbot.launch()