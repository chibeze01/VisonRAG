import os
import io
import base64
from dotenv import load_dotenv
from google import genai
from byaldi import RAGMultiModalModel
from PIL import Image

# Load environment variables from .env if it exists
load_dotenv()

# --- CONFIGURATION ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GOOGLE_API_KEY":
    print("Error: GEMINI_API_KEY not found or not set in .env file.")
    print("Please create a .env file with your GEMINI_API_KEY.")
    # Exit or handle as needed
    import sys
    sys.exit(1)

client = genai.Client(api_key=GEMINI_API_KEY)

# 1. Load the Index
index_name = "vision_rag_index"
if not os.path.exists(os.path.join(".byaldi", index_name)):
    print(f"Error: Index '{index_name}' not found. Please run index_pdf.py first.")
    import sys
    sys.exit(1)

print(f"Loading Index '{index_name}'...")
RAG = RAGMultiModalModel.from_index(index_name, device="cpu")

# 2. Define the Search Function
def visual_search(query_text, k=1):
    print(f"Searching for: {query_text}")
    results = RAG.search(query_text, k=k)
    
    if not results:
        return None, None

    # For this prototype, we'll take the top result
    result_item = results[0]
    base64_image = result_item.base64
    page_num = result_item.page_num
    
    # Decode to PIL Image
    image_data = base64.b64decode(base64_image)
    image = Image.open(io.BytesIO(image_data))
    
    return image, page_num

# 3. Define the Generation Function
def ask_gemini(image, question):
    print("Asking Gemini to analyze the retrieved page...")
    
    # Send image + question using the new google-genai API
    response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents=[
            "You are an expert technical assistant. Look at this document page and answer the question based ONLY on the provided image.",
            image,
            f"Question: {question}"
        ]
    )
    return response.text

def main():
    while True:
        user_query = input("\nEnter your question (or 'quit' to exit): ")
        if user_query.lower() in ['quit', 'exit', 'q']:
            break
            
        retrieved_image, page_num = visual_search(user_query)
        
        # save the retrived image
        retrieved_image.save("retrieved_image.png")

        if retrieved_image:
            print(f"Found relevant info on Page {page_num}. Analyzing...")
            # Optional: Display the image if in a notebook or GUI environment
            # retrieved_image.show() 
            
            answer = ask_gemini(retrieved_image, user_query)
            print("\n--- ANSWER ---")
            print(answer)
        else:
            print("No relevant info found in the document.")

if __name__ == "__main__":
    main()
