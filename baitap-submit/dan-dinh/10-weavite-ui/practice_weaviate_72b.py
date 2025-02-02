import os
import uuid
import pandas as pd
import weaviate
import gradio as gr
import kagglehub
from weaviate.classes.config import Configure, Property, DataType, Tokenization
from weaviate.util import generate_uuid5  # Generate a deterministic ID

vector_db_client = weaviate.connect_to_local(
    host="127.0.0.1",
    port=8080,
    grpc_port=50051,
    skip_init_checks=True
)

COLLECTION_NAME = "BookCollection"

def create_collection():
    """
    Create a new collection in Weaviate.
    Returns:
    - A weaviate.collections.Collection object representing the created collection.
    """
    # Create collection schema
    book_collection = vector_db_client.collections.create(
        name=COLLECTION_NAME,
        vectorizer_config=Configure.Vectorizer.text2vec_transformers(), # Use text2vec-transformers model to create vector embeddings
        properties=[
            Property(
                name="object_id",
                data_type=DataType.TEXT,
                tokenization=Tokenization.FIELD,
                skip_vectorization=True
            ),

            # Title property is vectorized and tokenized to lowercase
            Property(
                name="title",
                data_type=DataType.TEXT,
                vectorize_property_name=True,
                tokenization=Tokenization.LOWERCASE
            ),
            Property(
                name="author",
                data_type=DataType.TEXT,
                tokenization=Tokenization.FIELD,
            ),
            Property(
                name="description",
                data_type=DataType.TEXT,
                tokenization=Tokenization.WORD
            ),
            Property(
                name="grade",
                data_type=DataType.TEXT,
                tokenization=Tokenization.WORD
            ),
            Property(
                name="genre",
                data_type=DataType.TEXT,
                tokenization=Tokenization.FIELD
            ),
            Property(
                name="date",
                data_type=DataType.TEXT,
                tokenization=Tokenization.FIELD
            ),
            Property(
                name="intro",
                data_type=DataType.TEXT,
                tokenization=Tokenization.LOWERCASE
            ),

            # Skip vectorization for these properties
            Property(
                name="lexile",
                data_type=DataType.TEXT,
                skip_vectorization=True
            ),
            Property(
                name="path",
                data_type=DataType.TEXT,
                skip_vectorization=True
            ),
            Property(
                name="is_prose",
                data_type=DataType.INT,
                skip_vectorization=True
            ),
        ]
    )
    
    return book_collection

def generate_uuid5(data_row, namespace=uuid.NAMESPACE_DNS):
    """
    Generate a deterministic UUID (UUIDv5) based on the unique attributes of the data_row.
    Parameters:
    - data_row: A dictionary containing the unique attributes of the data row.
    - namespace: The namespace for the UUID. Default is uuid.NAMESPACE_DNS.
    Returns:
    - A string representing the generated UUID.
    """
    unique_string = f'{data_row["title"]}-{data_row["genre"]}'
    return str(uuid.uuid5(namespace, unique_string))

def insert_data(collection: weaviate.collections.Collection):
    """
    Insert data to Weaviate.
    Parameters:
    - collection: A weaviate.collections.Collection object representing the collection to insert data to.
    Returns:
    - None
    """
    # Download dataset latest version
    path = kagglehub.dataset_download("kononenko/commonlit-texts")

    # Print out the path
    print("Path to dataset files:", path)

    # Define the path to the CSV file
    file_path = os.path.join(path, "commonlit_texts.csv")

    # Read data from CSV file
    data = pd.read_csv(file_path)

    # Convert data to Weaviate objects
    sent_to_vector_db = data.to_dict(orient='records')
    total_records = len(sent_to_vector_db)
    print(f'Inserting data to Vector DB. Total records: {total_records}')

    # Insert data to Weaviate
    with collection.batch.dynamic() as batch:
        for data_row in sent_to_vector_db:
            print(f'Inserting record: {data_row["title"]}')

            # Generate UUIDv5
            obj_uuid = generate_uuid5(data_row)

            # Add object_id to data_row
            data_row["object_id"] = obj_uuid
            data_row["grade"] = str(data_row["grade"])    # Convert to string
            data_row["lexile"] = str(data_row["lexile"])    # Convert to string

            batch.add_object(
                properties=data_row,
                uuid=obj_uuid
            )
    print("Data is saved to Vector DB")

# Open DB connection
vector_db_client.connect()

print("DB is ready: {}".format(vector_db_client.is_ready()))

# Delete the collection if it exists
# vector_db_client.collections.delete(COLLECTION_NAME)

# Check if collection exists, if not create a new one and then insert data to the newly created collection
if not vector_db_client.collections.exists(COLLECTION_NAME):
    book_collection = create_collection()
    insert_data(book_collection)

def search_book(query: str):
    """
    Search for books in Weaviate.
    Parameters:
    - query (str): A string representing the search query.
    Returns:
    - results (list): A list of tuples representing the search results.
    """
    # Get the collection that has been created before
    book_collection = vector_db_client.collections.get(COLLECTION_NAME)

    # Hybrid search (combine text and vector search)
    repsonse_hybrid_search = book_collection.query.hybrid(
        query=query,
        # alpha = 0 -> search by text, alpha = 1 -> search by vector
        # alpha = 0.5 -> search by text and vector
        alpha=0.5,
        limit=15
    )

    results = []
    # Get the results with Title, Genre, Description properties
    for result in repsonse_hybrid_search.objects:
        book = result.properties
        book_tuple = (book['title'], book['genre'], book['description'])
        results.append(book_tuple)
    
    return results

with gr.Blocks(title="Book Search") as interface:
    query = gr.Textbox(label="Search book", placeholder="Book title, author name, genre, ...")
    search = gr.Button(value="Search")
    df = gr.Dataframe(label="Books", headers=["Title", "Genre", "Description"])

    # Handle events
    search.click(search_book, inputs=[query], outputs=[df])
    query.submit(search_book, inputs=[query], outputs=[df])

# Enables asynchronous processing with queue()
interface.queue().launch()

# Close DB connection
vector_db_client.close()
