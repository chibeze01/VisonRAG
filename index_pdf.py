from byaldi import RAGMultiModalModel
import os

def main():
    # 1. Load the Model
    # We use ColQwen2 because it's the current SOTA for visual retrieval
    print("Loading ColQwen2 model (this may take a while the first time)...")
    try:
        # Note: on Windows without a GPU, this might be slow or require specific torch config.
        # byaldi should handle basic CPU fallback if needed, but GPU is highly recommended.
        # Explicitly force CPU because CUDA is not available
        RAG = RAGMultiModalModel.from_pretrained("vidore/colqwen2-v1.0", device="cpu")

        # 2. Index the PDF
        # We'll use the sample.pdf provided by the user.
        pdf_path = "Quarterly Letters - Jun 2025.pdf"
        
        if not os.path.exists(pdf_path):
            print(f"Error: {pdf_path} not found. Please ensure the file exists in the directory.")
            return

        print(f"Indexing {pdf_path}...")
        # 'store_collection_with_index=True' saves the base64 images inside the index 
        # so you don't have to re-open the PDF later.
        RAG.index(
            input_path=pdf_path,
            index_name="vision_rag_index",
            store_collection_with_index=True,
            overwrite=True
        )

        print("Indexing complete. Saved to .byaldi/vision_rag_index")
    except Exception as e:
        print(f"An error occurred during indexing: {e}")

if __name__ == "__main__":
    main()
