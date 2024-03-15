import PyPDF2
import re
import pinecone

# Connect to Pinecone
pinecone.init(api_key="c916ccb6-974c-4aea-acbc-b1232fdf4005")
index_name = "pdf_index"

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

# Function to preprocess text
def preprocess_text(text):
    # Remove special characters, numbers, and extra spaces
    text = re.sub(r'\W+', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

# Function to convert text to vector
def text_to_vector(text):
    # You can use any embedding technique here, for demonstration, we'll just use simple count-based vectorization
    vector = [text.count(chr(i)) for i in range(256)]
    return vector

# Function to save vector to Pinecone
def save_vector_to_pinecone(vector, index_name):
    index = pinecone.Index(name=index_name)
    index.upsert(items=["pdf_document"], vectors=[vector])

# Main function
def main(pdf_path):
    # Extract text from PDF
    pdf_text = extract_text_from_pdf(pdf_path)
    
    # Preprocess text
    preprocessed_text = preprocess_text(pdf_text)
    
    # Convert text to vector
    vector = text_to_vector(preprocessed_text)
    
    # Save vector to Pinecone
    save_vector_to_pinecone(vector, index_name)
    
    print("PDF converted to vector and saved to Pinecone successfully.")

# Example usage
if __name__ == "__main__":
    pdf_path = "example.pdf"  # Path to your PDF file
    main(pdf_path)
