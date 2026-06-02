from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough,RunnableLambda

from app.utils.logger import get_logger,LoggerMixin
from app.config import get_settings
from app.core.vectorstore import VectorStore

settings = get_settings()
# RAG Prompt Template
RAG_PROMPT_TEMPLATE = """You are a helpful assistant. Answer the question based on the provided context.

If you cannot answer the question based on the context, say "I don't have enough information to answer that question."

Do not make up information. Only use the context provided.

Context:
{context}

Question: {question}

Answer:"""

def format_docs(docs: list[Document]) -> str:
    """Format documents for RAG prompt.
    
    Args:
        docs: List of Document objects

    Returns:
        Formatted documents as a single string
    
    
    """
    return "\n\n".join(doc.page_content for doc in docs)


class RAGChain(LoggerMixin):
    """Service for managing RAG chain operations."""

    def __init__(self, vector_store_service: VectorStore | None = None) :
        """Initialize RAG chain service.
        
        Args:
            vector_store_service: Vector store service instance
            
        """
        self.vector_store_service = vector_store_service or VectorStore()
        self.retriever = self.vector_store_service.get_retriever()
        
        self.evaluator=None

        self.llm=ChatGroq(
        model_name=settings.llm_model,
        temperature= settings.llm_temperature
        )

        self.prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
        
        self.chain = (
            {"context": self.retriever | format_docs,
            "question": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        logger.info(
            f"RAG chain initialized with LLM: {settings.llm_model},"
            f"retreval_k={settings.retrieval_k}"
        )

        @property
        def evaluator(self):
            """Get RAG evaluator."""
            if self.evaluator is None:
                from app.core.rag_evaluator import RAGEvaluator
                self.evaluator = RAGEvaluator()
            return self.evaluator
        
        def query(self,question:str) -> str:
            """Query the RAG chain.

            Args:
                question: Query text

            Returns:
                Answer text
            """
            self.logger.info(f"Processing Query: {question[:100]}....")
            try:
                answer=query.invoke(question)
                self.logger.info("Query processed successfully")
                return answer
            except Exception as e:
                self.logger.error(f"Error processing query: {e}")
                raise
        def stream(self,question:str) -> str:
            """Stream the RAG chain.

            Args:
                question: Query text

            Returns:
                Answer text
            """
            self.logger.info(f"Streaming Query: {question[:100]}....")
            try:
                answer=self.chain.stream(question)
                self.logger.info("Query processed successfully")
                return answer
            except Exception as e:
                self.logger.error(f"Error processing query: {e}")
                raise
        
        def query_with_sources(self,question:str) -> dict:
            """Query the RAG chain with sources.
            
            Args:
                question: Query text

            Returns:
                Dictionary with answer and sources
            """
            self.logger.info(f"Processing Query with sources: {question[:100]}....")
            try:
                anwer=self.chain.invoke(question)
                sources_docs=self.retriever.invoke(question)
                sources=[
                    {
                        "content": (
                            doc.page_content[:500]+"....[truncated]"
                        ) if len(doc.page_content)>500 else doc.page_content,
                        "metadata":doc.metadata
                        
                    }
                    for doc in sources_docs
                ]
                self.logger.info("Query processed successfully")
                return {"answer": answer, "sources": sources}
            except Exception as e:
                self.logger.error(f"Error processing query: {e}")
                raise
        async def aquery(self,question:str) -> str:
            """Async query the RAG chain.
            
            Args:
                question: Query text

            Returns:
                Answer text
            """
            self.logger.info(f"Processing Query: {question[:100]}....")
            try:
                answer=await self.chain.ainvoke(question)
                self.logger.info("Query processed successfully")
                return answer
            except Exception as e:
                self.logger.error(f"Error processing query: {e}")
                raise
                
        async def aquery_with_sources(self,question:str) -> dict:
            """Async query the RAG chain with sources.
            
            Args:
                question: Query text

            Returns:
                Dictionary with answer and sources
            """
            self.logger.info(f"Processing Query with sources: {question[:100]}....")
            try:
                answer=await self.chain.ainvoke(question)
                sources_docs=await self.retriever.ainvoke(question)
                sources=[
                    {
                        "content": (
                            doc.page_content[:500]+"....[truncated]"
                        ) if len(doc.page_content)>500 else doc.page_content,
                        "metadata":doc.metadata
                        
                    }
                    for doc in sources_docs
                ]
                self.logger.info("Query processed successfully")
                return {"answer": answer, "sources": sources}
            except Exception as e:
                self.logger.error(f"Error processing query: {e}")
                raise
        async def aquery_with_evaluation(self,question:str, include_sources: bool = True) -> dict:
            """Async query the RAG chain with evaluation.
            
            Args:
                question: Query text
                include_sources: Whether to include sources in the response

            Returns:
                Dictionary with answer and evaluation
            """
            self.logger.info(f"Processing Query with evaluation: {question[:100]}....")
            try:
                result=await self.aquery_with_sources(question)
                answer=result["answer"]
                sources= result['sources']

                contexts=[
                    source["content"] for source in sources
                ]
                try:
                    evaluation=await self.evaluator.evaluate(question,answer,contexts)
                    self.logger.info(
                    f"Evaluation completed - "
                    f"faithfulness={evaluation.get('faithfulness', 'N/A')}, "
                    f"answer_relevancy={evaluation.get('answer_relevancy', 'N/A')}"
                    )
                except Exception as e:
                    self.logger.error(f"Error evaluating query: {e}")
                    evaluation={
                    "faithfulness": None,
                    "answer_relevancy": None,
                    "evaluation_time_ms": None,
                    "error": str(e),
                }
                return {"answer": answer, "sources": sources, "evaluation": evaluation}
            except Exception as e:
                self.logger.error(f"Error processing query: {e}")
                raise
        def stream(self, question: str):
            """Stream the RAG chain with evaluation.
            
            Args:
                question: Query text

            Yields:
                Response chunks
            """
            self.logger.info(f"Streaming Query: {question[:100]}....")
            try:
                for chunk in self.chain.stream(question):
                    yield chunk
                self.logger.info("Query processed successfully")
            except Exception as e:
                self.logger.error(f"Error processing query: {e}")
                raise
