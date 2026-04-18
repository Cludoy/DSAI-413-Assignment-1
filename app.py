import streamlit as st
import base64
from retriever import WarframeQA

st.set_page_config(page_title="Warframe Knowledge Base", layout="wide", page_icon="🗡️")

@st.cache_resource
def load_qa_system():
    return WarframeQA()

qa_system = load_qa_system()

st.title("Warframe Multi-Modal RAG System")
st.markdown("Search across curated beginner's guide text, fan kit layouts, and strictly typed drop tables.")

query = st.text_input("Enter your Warframe query:", placeholder="E.g., What is the blueprint drop rate for Wisp?")

if query:
    with st.spinner("Searching multimodal context..."):
        retrieved_docs = qa_system.search(query, top_k=3)
    
    if not retrieved_docs:
        st.error("No relevant documents found. Please ensure `indexer.py` has been executed correctly and PDFs exist.")
    else:
        with st.spinner("Generating precision answer with Gemini Vision..."):
            answer = qa_system.generate_answer(query, retrieved_docs)
            
        st.subheader("Generated Response")
        st.info(answer)
        
        st.subheader("Source Visual Citations")
        
        # Determine number of columns based on retrieved objects dynamically bounded
        cols = st.columns(len(retrieved_docs))
        for idx, col in enumerate(cols):
            with col:
                doc = retrieved_docs[idx]
                st.markdown(f"**Doc Identifier:** {doc['doc_id']} | **Page:** {doc['page_num']} | **Score:** {doc['score']:.2f}")
                b64_img = doc['base64']
                st.image(base64.b64decode(b64_img), use_container_width=True, caption=f"Retrieved Source {idx+1}")
