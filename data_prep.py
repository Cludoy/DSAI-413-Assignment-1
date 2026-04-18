import json
import random
from collections import defaultdict
from pathlib import Path
from PIL import Image
import fitz  # PyMuPDF
from tqdm import tqdm

# Per-file page limits applied during manifest extraction and indexing.
# Populated at runtime with fankit PDFs; add overrides here if needed.
PAGE_LIMITS: dict[str, int | None] = {
    "drop_tables.pdf": 100,
    # fankit_<category>.pdf entries are added dynamically by compile_fankit_by_category()
}


def _safe_category_name(name: str) -> str:
    """Convert a folder name to a safe PDF filename stem."""
    return name.lower().replace(" ", "_").replace("/", "-")


def compile_fankit_by_category(
    fankit_dir: str,
    output_dir: str,
    images_per_pdf: int = 8,
    seed: int | None = None,
) -> list[str]:
    """
    Groups all images in fankit_dir by their top-level subfolder and produces
    one PDF per category, each containing a random sample of images_per_pdf images.

    Args:
        fankit_dir:    Path to the fankit root directory.
        output_dir:    Directory where output PDFs are written.
        images_per_pdf: Number of images randomly selected per category PDF.
        seed:          Random seed for reproducibility. If None, a new random
                       seed is chosen each run and printed so results can be
                       reproduced. Pass the same seed to get the same selection.

    Returns:
        List of output PDF file paths that were successfully created.
    """
    root_path = Path(fankit_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ── Discover all images ──────────────────────────────────────────────────
    all_images: list[Path] = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        all_images.extend(root_path.rglob(ext))

    if not all_images:
        print("No images found in fankit directory.")
        return []

    # ── Group by top-level subfolder → category ──────────────────────────────
    categories: dict[str, list[Path]] = defaultdict(list)
    for img_path in all_images:
        rel = img_path.relative_to(root_path)
        category = rel.parts[0] if len(rel.parts) > 1 else "General"
        categories[category].append(img_path)

    # ── Resolve seed ─────────────────────────────────────────────────────────
    if seed is None:
        seed = random.randint(0, 2**31)
    tqdm.write(f"Fankit random seed: {seed}  (reuse to reproduce this selection)")
    rng = random.Random(seed)

    created_pdfs: list[str] = []
    cat_bar = tqdm(sorted(categories.items()), desc="Fankit categories", unit="cat")

    for category, img_paths in cat_bar:
        cat_bar.set_postfix(category=category)
        safe_name = _safe_category_name(category)
        output_path = str(out_path / f"fankit_{safe_name}.pdf")

        # Random sample (or all images if fewer than images_per_pdf)
        sample = rng.sample(img_paths, min(images_per_pdf, len(img_paths)))
        tqdm.write(f"  [{category}] {len(img_paths)} images -> sampling {len(sample)} -> {Path(output_path).name}")

        pdf_images: list[Image.Image] = []
        for path in tqdm(sample, desc=f"    Loading", unit="img", leave=False):
            try:
                with Image.open(path) as img:
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    pdf_images.append(img.copy())
            except Exception as e:
                tqdm.write(f"    Skipped {path.name}: {e}")

        if not pdf_images:
            tqdm.write(f"  [{category}] No valid images loaded — skipping.")
            continue

        try:
            pdf_images[0].save(output_path, save_all=True, append_images=pdf_images[1:])
            tqdm.write(f"  [{category}] Saved -> {Path(output_path).name}")
            created_pdfs.append(output_path)
            # Register in PAGE_LIMITS (no cap for fankit PDFs)
            PAGE_LIMITS[Path(output_path).name] = None
        except Exception as e:
            tqdm.write(f"  [{category}] Error saving PDF: {e}")

    return created_pdfs


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
    print(f"\nMulti-modal manifest saved -> {manifest_path}")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main(seed: int | None = None):
    """
    Full data preparation pipeline:
      1. Compile fankit images → one PDF per category (randomized)
      2. Extract multi-modal metadata from all docs → JSON manifest

    Args:
        seed: Optional random seed for fankit image selection.
              Printed at runtime so you can reproduce a run.
    """
    docs_dir = Path("docs")
    docs_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Compile fan-kit images → one PDF per category
    fankit_dir = "fankit"
    if Path(fankit_dir).exists():
        print(f"\n── Fankit compilation ({'random seed=' + str(seed) if seed else 'random seed auto'}) ──")
        created = compile_fankit_by_category(
            fankit_dir=fankit_dir,
            output_dir=str(docs_dir),
            images_per_pdf=8,
            seed=seed,
        )
        print(f"Created {len(created)} fankit PDFs: {[Path(p).name for p in created]}")
    else:
        print(f"Warning: Fan Kit directory '{fankit_dir}' not found. Skipping fankit compilation.")

    # Step 2: Extract multi-modal metadata and write manifest
    print("\n── Multi-modal manifest extraction ──")
    build_multimodal_manifest(str(docs_dir))


if __name__ == "__main__":
    import sys
    # Optionally pass a seed as the first CLI argument: python data_prep.py 42
    cli_seed = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(seed=cli_seed)
