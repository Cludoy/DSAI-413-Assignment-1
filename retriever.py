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

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Executes Late Interaction (MaxSim) matrix comparisons bridging the text to 
        the document patch arrays locally natively outside Byaldi bindings. 
        """
        if not self.model or not self.doc_embeddings:
            return []
            
        with torch.no_grad():
            # Embed user question explicitly mapping to query tokens logic.
            query_inputs = self.processor.process_queries([query]).to(self.device)
            query_embedding = self.model(**query_inputs)[0].cpu() # shape: [num_query_tokens, hidden_dim]
            
            scores = []
            for doc_emb in self.doc_embeddings:
                # Late Interaction mechanism: MaxSim over document patches sum.
                # query_embedding: [N_q, D], doc_emb: [N_d, D]
                # einsum evaluates similarity matrix matching q to d iteratively
                sim_matrix = torch.einsum("qd,pd->qp", query_embedding, doc_emb)
                
                # Standard Colpali scoring format is the sum of maximum alignments
                score = sim_matrix.max(dim=1).values.sum().item()
                scores.append(score)
                
        # argsort highest first internally mapping the array
        top_indices = torch.tensor(scores).argsort(descending=True)[:top_k].tolist()
        
        retrieved_data = []
        for idx in top_indices:
            page_meta = self.pages[idx]
            retrieved_data.append({
                "doc_id": page_meta["doc_id"],
                "page_num": page_meta["page_num"],
                "base64": page_meta["base64"],
                "score": scores[idx]
            })
            
        return retrieved_data

    def generate_answer(self, query: str, retrieved_data: List[Dict[str, Any]]) -> str:
        """
        Passes the text question plus visual context to Gemini to anchor a generated response.
        """
        if not retrieved_data:
            return "Unable to perform generation. No multimodal context was returned from the index."

        contents = [
             "You are a helpful Warframe expert assistant. Answer the user's query utilizing ONLY the provided visual pages. "
             "The provided images may include rich text layouts, embedded fan-kit arts, and complex drop rate tables. "
             "Analyze the visual tables crossing rows and columns accurately. "
             "If the answer is NOT present visually in the context, explicitly indicate that you cannot answer it."
        ]
        
        for data in retrieved_data:
            img_bytes = base64.b64decode(data["base64"])
            contents.append(
                types.Part.from_bytes(
                     data=img_bytes,
                     mime_type="image/jpeg", 
                )
            )

        contents.append(f"User Query: {query}")
        
        try:
           response = self.client.models.generate_content(
                model='gemini-2.5-pro',
                contents=contents,
           )
           return response.text
        except Exception as e:
           return f"Error communicating with Gemini: {str(e)}"
