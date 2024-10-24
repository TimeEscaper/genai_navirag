from dataclasses import dataclass
from typing import List
from ragatouille import RAGPretrainedModel
from lib.models.llama import LlamaReaderQuantized
from lib.rag.knowledge_base import KnowledgeBase


@dataclass
class RAGOutput:
    lm_answer: str
    top_documents: List[str]


class RAGSystem:
    
    _PROMPT_IN_CHAT_FORMAT = [
        {
            "role": "system",
            "content": """Using the information contained in the context,
    give a comprehensive answer to the question.
    Respond only to the question asked, response should be concise and relevant to the question.
    Provide the number of the source document when relevant.
    If the answer cannot be deduced from the context, do not give an answer.""",
        },
        {
            "role": "user",
            "content": """Context:
    {context}
    ---
    Now here is the question you need to answer.

    Question: {question}""",
        },
    ]
    
    def __init__(self,
                 knowledge_base: KnowledgeBase,
                 reader_lm: LlamaReaderQuantized):
        self._db = knowledge_base
        self._reader_lm = reader_lm
        self._rag_prompt_template = reader_lm.tokenizer.apply_chat_template(
            RAGSystem._PROMPT_IN_CHAT_FORMAT,
            tokenize=False,
            add_generation_prompt=True
        )
        self._reranker = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
    
    def get_answer(self, query: str, rerank: bool, top_k: int = 5) -> RAGOutput:
        documents = self._db.search(query=query,
                                    top_k=top_k)
        if rerank:
            documents = self._reranker.rerank(query, documents, k=top_k)
            documents = [doc["content"] for doc in documents]

        context = "\nExtracted documents:\n"
        context += "".join(
            [f"Document {str(i)}:::\n" + doc for i, doc in enumerate(documents)]
        )

        final_prompt = self._rag_prompt_template.format(
            question=query, 
            context=context
        )
        
        answer = self._reader_lm.model(final_prompt)[0]["generated_text"]

        return RAGOutput(lm_answer=answer,
                         top_documents=documents)
    