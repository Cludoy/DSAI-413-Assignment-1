import json
from pathlib import Path
from PIL import Image
import fitz  # PyMuPDF
from tqdm import tqdm

# Per-file page limits (None = no limit)
PAGE_LIMITS: dict[str, int | None] = {
    "drop_tables.pdf": 100,
    "fankit_assets.pdf": None,
}


# ---------------------------------------------------------------------------
# Fan-Kit Compilation
# ---------------------------------------------------------------------------

def compile_fan_kit_images(fankit_dir: str, output_path: str, num_images: int = 8):
    """
    Recursively scans the provided fankit directory for image files,
    selects a limited subset, and concatenates them into a single PDF.
    """
    print(f"Scanning {fankit_dir} for visual assets...")
    root_path = Path(fankit_dir)
    image_paths = []

    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_paths.extend(list(root_path.rglob(ext)))

    if not image_paths:
        print("No images found in fankit directory.")
        return

    print(f"Found {len(image_paths)} images. Processing subset of {num_images}...")
    selected_paths = image_paths[:num_images]
    pdf_images = []

    for path in selected_paths:
        try:
            with Image.open(path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                pdf_images.append(img.copy())
                print(f"  Loaded {path.name}")
        except Exception as e:
            print(f"  Failed to process {path.name}: {e}")

    if pdf_images:
        try:
            pdf_images[0].save(output_path, save_all=True, append_images=pdf_images[1:])
            print(f"Successfully compiled Fan Kit PDF → {output_path}")
        except Exception as e:
            print(f"Error saving Fan Kit PDF: {e}")


# ---------------------------------------------------------------------------
# Multi-Modal Metadata Extraction
# ---------------------------------------------------------------------------

def _classify_page(text: str, table_count: int, image_count: int) -> str:
    """Heuristically classify the dominant content type of a page."""
    if table_count > 0 and len(text.strip()) > 50:
        return "table"
    if image_count > 0 and len(text.strip()) < 100:
        return "image"
    if len(text.strip()) > 200:
        return "text"
    return "mixed"


def extract_multimodal_metadata(pdf_path: str, max_pages: int | None = None) -> list[dict]:
    """
    Extracts three modalities from every page of a PDF:
      - Text  : plain text layer via PyMuPDF text extraction
      - Tables: structured row/column data via PyMuPDF find_tables()
      - Charts/Images: embedded image metadata (dimensions, count, positions)

    Args:
        pdf_path:  Path to the PDF file.
        max_pages: If set, only the first N pages are processed.

    Returns a list of per-page dicts ready to be merged into the index.
    """
    doc = fitz.open(pdf_path)
    total = len(doc)
    if max_pages is not None:
        total = min(total, max_pages)
    pages_meta = []

    pdf_name = Path(pdf_path).name
    for page_num, page in enumerate(
        tqdm(doc, total=total, desc=f"  {pdf_name}", unit="pg"), start=1
    ):
        if max_pages is not None and page_num > max_pages:
            break
        # ── 1. Text extraction ──────────────────────────────────────────────
        text = page.get_text("text").strip()

        # ── 2. Table extraction ─────────────────────────────────────────────
        extracted_tables = []
        try:
            finder = page.find_tables()
            for table in finder.tables:
                rows = table.extract()
                if rows:
                    # First non-None row is treated as headers
                    headers = [str(c) if c else "" for c in rows[0]]
                    data_rows = [
                        [str(c) if c else "" for c in row]
                        for row in rows[1:]
                    ]
                    extracted_tables.append({
                        "headers": headers,
                        "rows": data_rows
                    })
        except Exception as e:
            print(f"    Table extraction failed on page {page_num}: {e}")

        # ── 3. Embedded image / chart metadata ──────────────────────────────
        image_metadata = []
        for img_info in page.get_images(full=True):
            xref, smask, width, height = img_info[0], img_info[1], img_info[2], img_info[3]
            image_metadata.append({
                "xref": xref,
                "width_px": width,
                "height_px": height,
                "aspect_ratio": round(width / height, 3) if height else None
            })

        # ── 4. Page type classification ─────────────────────────────────────
        page_type = _classify_page(text, len(extracted_tables), len(image_metadata))

        pages_meta.append({
            "page_num": page_num,
            "page_type": page_type,
            "text": text,
            "tables": extracted_tables,
            "image_metadata": image_metadata,
        })

    doc.close()
    return pages_meta


def build_multimodal_manifest(docs_dir: str = "docs", manifest_path: str = "docs/multimodal_manifest.json"):
    """
    Iterates all PDFs in docs_dir, extracts multi-modal metadata for every page,
    and writes a consolidated JSON manifest alongside the PDFs.

    The manifest is consumed by indexer.py to enrich each page's stored metadata.
    """
    docs_path = Path(docs_dir)
    pdf_files = list(docs_path.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDFs found in '{docs_dir}'.")
        return

    manifest = {}
    outer_bar = tqdm(pdf_files, desc="PDFs", unit="file")
    for pdf_file in outer_bar:
        outer_bar.set_postfix(file=pdf_file.name)
        max_pages = PAGE_LIMITS.get(pdf_file.name)  # None = no limit
        pages = extract_multimodal_metadata(str(pdf_file), max_pages=max_pages)
        manifest[pdf_file.name] = pages

        # Per-PDF summary
        total_tables = sum(len(p["tables"]) for p in pages)
        total_images = sum(len(p["image_metadata"]) for p in pages)
        total_text_chars = sum(len(p["text"]) for p in pages)
        limit_str = f" (capped at {max_pages}pg)" if max_pages else ""
        tqdm.write(
            f"  {pdf_file.name}{limit_str}: {len(pages)} pages | "
            f"Tables: {total_tables} | Images: {total_images} | Text chars: {total_text_chars}"
        )

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"\nMulti-modal manifest saved → {manifest_path}")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main():
    docs_dir = Path("docs")
    docs_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Compile fan-kit images → PDF
    fankit_dir = "fankit"
    fankit_output_path = str(docs_dir / "fankit_assets.pdf")
    if Path(fankit_dir).exists():
        compile_fan_kit_images(fankit_dir, fankit_output_path, num_images=8)
    else:
        print(f"Warning: Fan Kit directory '{fankit_dir}' not found. Skipping fan kit compilation.")

    # Step 2: Extract multi-modal metadata and write manifest
    build_multimodal_manifest(str(docs_dir))


if __name__ == "__main__":
    main()
