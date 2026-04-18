import os
import torch
import base64
import io
import fitz  # PyMuPDF
from pathlib import Path
from PIL import Image
from colpali_engine.models import ColIdefics3, ColIdefics3Processor

def build_index(docs_dir: str = "docs", output_index: str = "docs/colsmol_index.pt"):
    """
    Reads PDFs from docs_dir, encodes them directly via ColIdefics3 (colSmol-500M),
    and saves the raw tensors alongside base64 patches for generation.
    """
    docs_path = Path(docs_dir)
    if not list(docs_path.glob("*.pdf")):
        print(f"No PDFs found in {docs_dir}.")
        return

    print("Initializing vidore/colSmol-500M via native ColIdefics3...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model and processor natively bridging the architectural restriction
    model_name = "vidore/colSmol-500M"
    model = ColIdefics3.from_pretrained(model_name).to(device).eval()
    processor = ColIdefics3Processor.from_pretrained(model_name)

    # Document store mechanism
    index_data = {
        "doc_embeddings": [],
        "pages": [] # Storing dicts of base64 and identifying tracking
    }
    
    print(f"Indexing PDFs inside '{docs_dir}'...")
    with torch.no_grad():
        for pdf_file in docs_path.glob("*.pdf"):
            print(f"Processing {pdf_file.name}...")
            
            # Load PDF via PyMuPDF internally removing Poppler strict dependencies
            doc = fitz.open(pdf_file)
            images = []
            for page in doc:
                pix = page.get_pixmap(dpi=150)
                img = Image.open(io.BytesIO(pix.tobytes("jpeg")))
                images.append(img)
            doc.close()
            
            # Avoid out of memory by batching locally
            batch_size = 4
            for i in range(0, len(images), batch_size):
                batch_imgs = images[i : i + batch_size]
                
                # Pre-process image to Base64 mapping for later Gemini evaluation
                for idx_b, img in enumerate(batch_imgs):
                    buffered = io.BytesIO()
                    # Enforce RGB to avoid channel corruptions
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img.save(buffered, format="JPEG")
                    b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                    
                    index_data["pages"].append({
                        "doc_id": pdf_file.name,
                        "page_num": i + idx_b + 1,
                        "base64": b64
                    })

                # ColPali engine logic extracting deep patch arrays
                batch_inputs = processor.process_images(batch_imgs).to(device)
                batch_embeddings = model(**batch_inputs)
                
                # Split the batch correctly and detach to CPU RAM
                for embedding in list(torch.unbind(batch_embeddings.cpu())):
                    index_data["doc_embeddings"].append(embedding)

    print(f"Saving compiled Index tensor block to {output_index}")
    torch.save(index_data, output_index)
    print("Done!")

if __name__ == "__main__":
    build_index()
