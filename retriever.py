import os
import torch
import base64
from typing import List, Dict, Any
from google import genai
from google.genai import types
from dotenv import load_dotenv
from colpali_engine.models import ColIdefics3, ColIdefics3Processor

load_dotenv()

class WarframeQA:
    def __init__(self, index_path: str = "docs/colsmol_index.pt"):
        """
        Loads the pre-saved PyTorch index dictionary offline and mounts the Gemini integration.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Loading local ColSmol tensor index...")
        try:
            self.index_data = torch.load(index_path, map_location="cpu")
            self.pages = self.index_data["pages"]
            # Collate the list of N [patch, hidden_dim] arrays.
            self.doc_embeddings = self.index_data["doc_embeddings"]
            
            print("Mounting colSmol query encoder...")
            model_name = "vidore/colSmol-500M"
            self.model = ColIdefics3.from_pretrained(model_name).to(self.device).eval()
            self.processor = ColIdefics3Processor.from_pretrained(model_name)
            
        except Exception as e:
            print(f"Error loading index '{index_path}': {e}. Ensure indexer.py is completed.")
            self.model = None
            
        print("Initializing Gemini API Client...")
        if not os.getenv("GEMINI_API_KEY"):
             print("WARNING: GEMINI_API_KEY missing from environment payload!")
        self.client = genai.Client()

    def _keyword_score(self, query: str, text: str) -> float:
        """
        Lightweight keyword overlap score: fraction of query tokens found in the page text.
        Used to boost pages that have strong textual/tabular coverage of the query.
        """
        if not text:
            return 0.0
        query_tokens = set(query.lower().split())
        text_lower = text.lower()
        matches = sum(1 for tok in query_tokens if tok in text_lower)
        return matches / max(len(query_tokens), 1)

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval combining:
          1. ColPali Late Interaction (MaxSim) — visual/semantic similarity
          2. Keyword overlap score — boosts pages with strong text/table coverage

        Final score = 0.75 * visual_score_normalised + 0.25 * keyword_score
        """
        if not self.model or not self.doc_embeddings:
            return []

        with torch.no_grad():
            query_inputs = self.processor.process_queries([query]).to(self.device)
            query_embedding = self.model(**query_inputs)[0].cpu()

            visual_scores = []
            for doc_emb in self.doc_embeddings:
                sim_matrix = torch.einsum("qd,pd->qp", query_embedding, doc_emb)
                score = sim_matrix.max(dim=1).values.sum().item()
                visual_scores.append(score)

        # Normalise visual scores to [0, 1]
        vis_tensor = torch.tensor(visual_scores, dtype=torch.float32)
        vis_min, vis_max = vis_tensor.min(), vis_tensor.max()
        if vis_max > vis_min:
            vis_norm = ((vis_tensor - vis_min) / (vis_max - vis_min)).tolist()
        else:
            vis_norm = [1.0] * len(visual_scores)

        # Keyword scores from extracted text layer
        kw_scores = [
            self._keyword_score(query, page.get("text", ""))
            for page in self.pages
        ]

        # Hybrid fusion
        hybrid_scores = [
            0.75 * v + 0.25 * k
            for v, k in zip(vis_norm, kw_scores)
        ]

        # argsort highest first
        top_indices = torch.tensor(hybrid_scores).argsort(descending=True)[:top_k].tolist()

        retrieved_data = []
        for idx in top_indices:
            page_meta = self.pages[idx]
            retrieved_data.append({
                "doc_id":        page_meta["doc_id"],
                "page_num":      page_meta["page_num"],
                "base64":        page_meta["base64"],
                "page_type":     page_meta.get("page_type", "unknown"),
                "text":          page_meta.get("text", ""),
                "tables":        page_meta.get("tables", []),
                "image_metadata":page_meta.get("image_metadata", []),
                "score":         hybrid_scores[idx],
                "visual_score":  visual_scores[idx],
                "keyword_score": kw_scores[idx],
            })

        return retrieved_data

    def generate_answer(self, query: str, retrieved_data: List[Dict[str, Any]]) -> str:
        """
        Passes the text question plus multi-modal context to Gemini:
          - Visual page images (for layout, charts, and visual tables)
          - Extracted text blocks (for precise textual answers)
          - Structured table data (for accurate numeric/row lookups)
          - Chart/image metadata (dimensions and counts per page)
        """
        if not retrieved_data:
            return "Unable to perform generation. No multimodal context was returned from the index."

        # ── System instruction ───────────────────────────────────────────────
        system_prompt = (
            "You are a knowledgeable Warframe expert assistant. "
            "You have been given multi-modal context from a curated knowledge base, which includes: "
            "(1) visual page images, (2) extracted page text, and (3) structured table data. "
            "Answer the user's query using ALL available context — visual, textual, and tabular. "
            "When answering questions about drop rates or item tables, prefer the structured table data "
            "for precision and cross-reference with the visual image. "
            "If the answer is NOT present in any of the provided context, explicitly say so."
        )

        contents = [system_prompt]

        for i, data in enumerate(retrieved_data):
            page_label = f"[Source {i+1}: {data['doc_id']} — Page {data['page_num']} ({data.get('page_type','unknown')} page)]"
            contents.append(page_label)

            # 1. Visual page image
            img_bytes = base64.b64decode(data["base64"])
            contents.append(
                types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg")
            )

            # 2. Extracted text (if present and substantive)
            text = data.get("text", "").strip()
            if text:
                contents.append(f"--- Extracted Text (page {data['page_num']}) ---\n{text[:3000]}")  # cap at 3k chars

            # 3. Structured table data (if present)
            tables = data.get("tables", [])
            if tables:
                table_str_parts = []
                for t_idx, table in enumerate(tables):
                    headers = " | ".join(table.get("headers", []))
                    rows = "\n".join(" | ".join(row) for row in table.get("rows", [])[:30])  # cap at 30 rows
                    table_str_parts.append(f"Table {t_idx+1}:\n{headers}\n{rows}")
                contents.append(f"--- Structured Tables (page {data['page_num']}) ---\n" + "\n\n".join(table_str_parts))

            # 4. Chart/image metadata
            img_meta = data.get("image_metadata", [])
            if img_meta:
                meta_desc = "; ".join(
                    f"img_{j+1}: {m['width_px']}×{m['height_px']}px (AR={m['aspect_ratio']})"
                    for j, m in enumerate(img_meta)
                )
                contents.append(f"--- Embedded Chart/Image Metadata (page {data['page_num']}) ---\n{meta_desc}")

        contents.append(f"\nUser Query: {query}")

        try:
            response = self.client.models.generate_content(
                model='gemini-2.5-pro',
                contents=contents,
            )
            return response.text
        except Exception as e:
            return f"Error communicating with Gemini: {str(e)}"
