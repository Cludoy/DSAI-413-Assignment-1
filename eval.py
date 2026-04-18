import os
from glob import glob
from retriever import WarframeQA

# Evaluation suite scoped to Beginner elements and Drop Tables specifically
BENCHMARK_QUESTIONS = [
    # Table Extraction specifically (Drop Tables and Sortie/Relic structure)
    "Examine the drop tables. What is the listed drop configuration or reward pool for Sorties?",
    "Identify any items in the provided tables that have an 'Uncommon' or 'Legendary' rarity. What are they?",
    "Based on the tables, what is the exact numerical drop chance attached to Endo rewards or Anasa Ayatan Sculptures?",
    "Is there any Relic or Forma specific drop listed in the extracted data frame snippet? Provide the parameters.",
    
    # Text Analysis (Beginner's Guide)
    "According to the beginner's guide text, what are the three starter Warframes available when you first begin the game?",
    "Explain the 'Bullet Jump' mechanic as described in the movement section of the guide.",
    "What is the recommended method in the guide for a beginner to level up their Mastery Rank efficiently?",
    
    # Image / Visual verification (Images compiled from Fan Kit)
    "Analyze the visual details of the 'Harrow Deluxe' artwork. What distinct thematic elements are present in his design?",
    "Observe the 'CorpusEnvironmentSuit' and 'CorpusRobotAssembly' images. How is the Corpus aesthetic visually distinct from the Grineer aesthetics shown in 'Orokin Era Grineer'?",
    "Look at the 'TennoGuns' asset image. Describe the geometric styling and material textures on the featured weapon."
]

def run_evaluation(index_name: str = "warframe_index"):
    """
    Executes the 10 domain-specific benchmark questions against the Multi-Modal pipeline
    and logs the generation performance natively.
    """
    print(f"Initializing QA Pipeline for Evaluation Suite...")
    qa_system = WarframeQA(index_path="docs/colsmol_index.pt")
    
    if qa_system.model is None:
        print("Indexer is not set up correctly. Please deploy indexer.py first on the curated PDFs.")
        return

    results = []
    
    for i, question in enumerate(BENCHMARK_QUESTIONS, start=1):
        print(f"\n--- [Eval {i}/10] ---")
        print(f"Q: {question}")
        
        # 1. Retrieval
        retrieved_docs = qa_system.search(question, top_k=2)
        if not retrieved_docs:
            print("A: [FAIL] No documents retrieved.")
            continue
            
        # 2. Generation
        answer = qa_system.generate_answer(question, retrieved_docs)
        print(f"A: {answer}")
        print(f"Context sourced from {len(retrieved_docs)} pages.")

if __name__ == "__main__":
    run_evaluation()
