"""
Test script to verify scanned PDF OCR functionality
"""
import sys
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent))

from app.rag import load_file_to_documents

def test_scanned_pdf(pdf_path: str):
    """Test loading a scanned PDF with OCR"""
    print(f"\n{'='*60}")
    print(f"Testing Scanned PDF OCR")
    print(f"{'='*60}")
    print(f"File: {pdf_path}")
    
    try:
        # Test if the file exists
        if not Path(pdf_path).exists():
            print(f"❌ Error: File not found: {pdf_path}")
            return
        
        print("\n📄 Loading document with OCR fallback...")
        documents = load_file_to_documents(pdf_path)
        
        # Check results
        if not documents:
            print("❌ No documents returned")
            return
        
        # Check for errors
        if documents[0].metadata.get("error", False):
            print(f"❌ Error: {documents[0].page_content}")
            return
        
        # Success!
        print(f"\n✅ Successfully processed scanned PDF!")
        print(f"📊 Pages extracted: {len(documents)}")
        print(f"\n{'='*60}")
        print("Sample from first page:")
        print(f"{'='*60}")
        first_page_text = documents[0].page_content
        preview = first_page_text[:500] if len(first_page_text) > 500 else first_page_text
        print(preview)
        if len(first_page_text) > 500:
            print(f"\n... (total {len(first_page_text)} characters)")
        
        print(f"\n{'='*60}")
        print("Metadata:")
        print(f"{'='*60}")
        for key, value in documents[0].metadata.items():
            print(f"  {key}: {value}")
        
        print(f"\n✅ Test PASSED!")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Test with the problematic file
    test_file = r"uploads\020DAFGS4 Business law 2022 2023.pdf"
    
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
    
    test_scanned_pdf(test_file)
