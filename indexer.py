import os
import json
import torch
import base64
import io
import fitz  # PyMuPDF
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from colpali_engine.models import ColIdefics3, ColIdefics3Processor
from data_prep import PAGE_LIMITS

def build_index(docs_dir: str = "docs", output_index: str = "docs/colsmol_index.pt",
                manifest_path: str = "docs/multimodal_manifest.json"):
    """
    Reads PDFs from docs_dir, encodes them via ColIdefics3 (colSmol-500M),
    and saves the raw tensors alongside base64 patches and multi-modal metadata
    (text, tables, chart/image info) for enriched generation.

    Run data_prep.py first to build the multimodal_manifest.json.
    """
    docs_path = Path(docs_dir)
    if not list(docs_path.glob("*.pdf")):
        print(f"No PDFs found in {docs_dir}.")
        return

    # Load the multi-modal manifest produced by data_prep.py
    manifest: dict = {}
    if Path(manifest_path).exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        print(f"Loaded multi-modal manifest: {len(manifest)} PDFs with pre-extracted text/tables/images.")
    else:
        print(f"Warning: manifest not found at '{manifest_path}'. "
              "Run data_prep.py first for full multi-modal coverage. Continuing with visual-only mode.")

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
    pdf_files = sorted(docs_path.glob("*.pdf"))
    with torch.no_grad():
        for pdf_file in tqdm(pdf_files, desc="PDFs", unit="file"):
            max_pages = PAGE_LIMITS.get(pdf_file.name)  # None = no limit
            limit_str = f" (capped at {max_pages} pages)" if max_pages else ""
            tqdm.write(f"Processing {pdf_file.name}{limit_str}...")

            # Load PDF pages as images via PyMuPDF
            doc = fitz.open(pdf_file)
            images = []
            total_pages = min(len(doc), max_pages) if max_pages else len(doc)
            for page_idx, page in enumerate(
                tqdm(doc, total=total_pages, desc=f"  Rendering {pdf_file.name}", unit="pg", leave=False)
            ):
                if max_pages is not None and page_idx >= max_pages:
                    break
                pix = page.get_pixmap(dpi=150)
                img = Image.open(io.BytesIO(pix.tobytes("jpeg")))
                images.append(img)
            doc.close()
            
            # Avoid out of memory by batching locally
            batch_size = 4
            num_batches = (len(images) + batch_size - 1) // batch_size
            for i in tqdm(
                range(0, len(images), batch_size),
                total=num_batches,
                desc=f"  Encoding {pdf_file.name}",
                unit="batch",
                leave=False,
            ):
                batch_imgs = images[i : i + batch_size]
                
                # Pre-process image to Base64 mapping for later Gemini evaluation
                for idx_b, img in enumerate(batch_imgs):
                    buffered = io.BytesIO()
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img.save(buffered, format="JPEG")
                    b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                    
                    page_num = i + idx_b + 1

                    # Look up multi-modal metadata from manifest (keyed by filename + page_num)
                    pdf_manifest_pages = manifest.get(pdf_file.name, [])
                    page_meta = next(
                        (p for p in pdf_manifest_pages if p["page_num"] == page_num),
                        {}
                    )

                    index_data["pages"].append({
                        "doc_id":         pdf_file.name,
                        "page_num":        page_num,
                        "base64":          b64,
                        # Multi-modal enrichment fields
                        "page_type":       page_meta.get("page_type", "unknown"),
                        "text":            page_meta.get("text", ""),
                        "tables":          page_meta.get("tables", []),
                        "image_metadata":  page_meta.get("image_metadata", []),
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
