#################################################
# Author:   Dan Dinh                            #
# Date:     2024-11-28                          #
# Exercise: 2 with Together AI                  #
# Question: 4 with document translation         #
#################################################

import os
from groq import Groq
from dotenv import load_dotenv
import fitz
from reportlab.pdfgen import canvas
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.lib.pagesizes import letter

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variables
api_key = os.getenv('API_KEY')

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file and return it as a list of strings.
    Parameters:
    - pdf_path: Path to the PDF file
    Returns:
    - pdf_content: List of text strings
    """
    try:
        # Open the PDF file
        pdf_document = fitz.open(pdf_path)
        print(f"Number of pages: {pdf_document.page_count}")
        
        # Define a list to store the extracted text
        pdf_content = []
        
        # Extract text from each page
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]  # Get page
            text = page.get_text()  # Extract text
            pdf_content.append(text)
            print(f"Page {page_num + 1}:")
            print(text)
            # Only extract text from the first 4 pages for testing purposes
            if page_num > 4:
                break

        # Close the document
        pdf_document.close()
        
        return pdf_content
    except Exception as e:
        print(f"An error occurred: {e}")


def get_response(messages):
    """
    Get the response from the chat completion API.
    Parameters:
    - messages: List of messages to send to the API
    Outputs:
    - response: The response from the API
    """
    # Create Groq client
    client = Groq(
        api_key=api_key,
    )

    # Create chat completion
    chat_completion = client.chat.completions.create(        
        messages=messages,
        model="llama-3.3-70b-versatile",
        max_tokens=8192,
        stream=True,
    )

    return_message = ""
    # Print response from chat completion
    for chunk in chat_completion:
        response = chunk.choices[0].delta.content or ""
        print(response, end="")
        return_message += response

    print("\n")
    return return_message

def wrap_text(text, width, font_size, pdf):
    """
    Word-wrap text to fit within a specified width.
    Parameters:
    - text: The text to wrap
    - width: The maximum width of the wrapped text
    - font_size: The font size of the text
    Outputs:
    - lines: A list of wrapped lines
    """
    words = text.split()  # Split the text into words
    lines = []  # List to store wrapped lines
    current_line = ""  # Temp variable to build the current line

    for word in words:
        # Check the width of the current line + the new word
        if pdf.stringWidth(current_line + word, 'Arial', font_size) <= width:
            current_line += word + " "
        else:
            lines.append(current_line.strip())  # Save the line
            current_line = word + " "  # Start a new line with the current word

    # Append the last line
    if current_line:
        lines.append(current_line.strip())

    return lines

def summarize_text(messages, max_words=3500):
    """
    Summarize the provided text in English while ensuring the following:
    Parameters:
    - messages: List of messages to summarize
    - max_words: Maximum number of words in the summary
    Outputs:
    - None
    """
    # Count total words in both 'role' and 'content' fields
    total_words = sum(
        len(str(value).split())  # Split and count words for each value
        for message in messages
        for key, value in message.items()  # Iterate through both 'role' and 'content'
    )

    # Check if the total words exceed the maximum limit
    if total_words > max_words:
        summarized_prompt = f"""
            You are a professional summarizer. Summarize the provided text in English while ensuring the following:
                1. Only output the results, no need to write the introduction or conclusion.
                2. Make sure the summary is coherent, captures the essence of the article.
                3. Make sure the summary ends with a full stop and less than 500 words.
            """

        summarized_message = []

        # Set summarized_message by excluding the third and fourth items from messages
        summarized_message = messages[:2] + messages[4:]

        # Add the initial prompt to the summarized message
        summarized_message.append({"role": "user", "content": summarized_prompt})

        # Get response from the chat completion API
        return_message = get_response(summarized_message)

        # Add response to summarized message array
        summarized_message.append({"role": "assistant", "content": return_message})

        # Build the new replacement list
        replacement = [
            {"role": "user", "content": "Here is the summarized text"},
            summarized_message[-1]  # Get the last item of summarized_message
        ]

        # Construct the new messages list
        messages[:] = replacement + messages[2:4]  # Add replacement items at the top and retain the first two original items

def create_pdf_from_text(pdf_content, output_pdf_path):
    """
    Create a PDF document from a list of text strings.
    Parameters:
    - pdf_content: List of text strings
    - output_pdf_path: Path to save the PDF document
    Outputs:
    - None
    """
    # Register a font that supports Unicode (e.g., Vietnamese diacritics)
    pdfmetrics.registerFont(TTFont('Arial', 'Arial.ttf'))  # Use a Unicode font
    
    # Create a new PDF document
    pdf = canvas.Canvas(output_pdf_path, pagesize=letter)
    pdf.setFont('Arial', 12)  # Set font and size
    
    # Define margins and page width
    x_margin = 50
    y_margin = 750
    page_width = 500  # Effective width for text (reduce for margins)
    line_height = 14

    # Translate and write each line of text to the PDF  
    for text in pdf_content:
        # Summarize the text if it exceeds the word limit (tokens size)
        summarize_text(messages)

        # Add user question to messages array
        messages.append({"role": "user", "content": text})
        
        # Get response from the chat completion API
        return_message = get_response(messages)

        # Word-wrap the text
        wrapped_lines = wrap_text(return_message, page_width, 12, pdf)

        for line in wrapped_lines:
            # Check if we need to start a new page
            if y_margin <= 50:
                pdf.showPage()  # Create a new page
                pdf.setFont('Arial', 12)  # Reset the font on the new page
                y_margin = 750  # Reset the y-margin for the new page
            
            # Draw the text on the PDF
            pdf.drawString(x_margin, y_margin, line)
            y_margin -= line_height  # Move down for the next line

        # Add response to messages array
        messages.append({"role": "assistant", "content": return_message})

    # Save the PDF to the specified path
    pdf.save()
    print(f"PDF saved successfully as: {output_pdf_path}")

# Get user input for the file path
pdf_path = input("Enter the file path: ")

# Extract text from the PDF
pdf_content = extract_text_from_pdf(pdf_path)

# Define the prompt for the chat completion
initial_prompt = f"""
    You are a professional translator specializing in Vietnamese and domain knowledge of the provided document. Translate the provided text into Vietnamese while ensuring the following:
        1. Only output the results, no need to write the introduction or conclusion.
        2. Use only valid and accurate Vietnamese characters, avoiding any incorrect or non-Vietnamese symbols or characters.
        3. Ensure the translation accurately conveys the original meaning and context, focusing on clarity and professionalism.
        4. Make sure correctly translate technical terms and context-specific language into Vietnamese to ensure they are understood by readers in the field.
        5. Avoid adding unrelated notes or comments; provide only the fully translated Vietnamese text.
        6. Pay close attention to grammar, syntax, and terminology to ensure the translation is coherent and error-free.
    """

messages = []

# Initialize the messages array with the initial prompt
messages.extend([
    {"role": "user", "content": "Can you help me translate this document into Vietnamese?"},
    {"role": "assistant", "content": "Yes, I am here to help."},
    {"role": "user", "content": initial_prompt},
    {"role": "assistant", "content": "I am waiting for the document content."}
])

# Get the current folder path
current_folder_path = os.path.dirname(os.path.realpath(__file__))

# Get the file name and extension
file_name_with_extension = os.path.basename(pdf_path)

# Define the output file path with the translated file name
translated_file = os.path.join(current_folder_path, f"Translated_{file_name_with_extension}")

# Create a PDF document with the translated text
create_pdf_from_text(pdf_content, translated_file)