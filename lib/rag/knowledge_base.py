from __future__ import annotations
from typing import List, Dict
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.docstore.document import Document as LangchainDocument


def create_embedding_model(device: str = "cuda") -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name="thenlper/gte-small",
        multi_process=True,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_database(places_texts: List[str],
                   embedding_model: HuggingFaceEmbeddings) -> FAISS:
    documents = [LangchainDocument(page_content=text) for text in places_texts]
    database = FAISS.from_documents(
        documents, 
        embedding_model, 
        distance_strategy=DistanceStrategy.COSINE
    )
    return database


class KnowledgeBase:
    
    def __init__(self,
                 faiss_db: FAISS):
        self._db = faiss_db
        
    @staticmethod
    def build(source_data: List[Dict[str, str]],
              use_captions: bool,
              use_graphs: bool,
              device: str = "cuda") -> KnowledgeBase:
        if not use_captions and not use_graphs:
            raise ValueError(f"Captions and/or graphs must be used")
        
        embedding_model = HuggingFaceEmbeddings(
            model_name="thenlper/gte-small",
            multi_process=True,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )
        
        texts = []
        for item in source_data:
            text = f"Image ID: {item['id']}\n"
            if use_captions:
                text = text + f"Image description:\n{item['caption']}\n"
            if use_graphs:
                text = text + f"Scene graph:\n{item['graph']}"
            texts.append(text)
            
        documents = [LangchainDocument(page_content=text) for text in texts]
        database = FAISS.from_documents(
            documents, 
            embedding_model, 
            distance_strategy=DistanceStrategy.COSINE
        )
        
        return KnowledgeBase(database)

    def search(self, query: str, top_k: int = 5) -> List[str]:
        retrieved = self._db.similarity_search(query=query, 
                                               k=top_k)
        return [e.page_content for e in retrieved]
