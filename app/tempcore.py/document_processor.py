"""Document processing module for loading and chunking documents."""

import tempfile
from pathlib import Path
from typing import BinaryIO
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.utils.logger import LoggerMixin
from langchain_community.document_loaders import (
    CSVLoader,
    PyPDFLoader,
    TextLoader,
)
from app.config import get_settings
from app.utils.logger import get_logger

class DocumentProcessor(LoggerMixin):
    """Process documents by loading and chunking them for RAG."""
    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".csv"}
    def __init__(self, chunk_size: int | None = None, chunk_overlap: int |None = None):
        """
        Initialize document processor.
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        settings = get_settings()
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ".", " "],
        )
        self.logger.info(
            f"DocumentProcessor initialized with chunk_size={self.chunk_size}, "
            f"chunk_overlap={self.chunk_overlap}"
        )
    def  load_pdf(self, file_path : str |Path) -> list[Document]:
        """
        Load a PDF document from a file path.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of Document objects
        """
        try:
            file_path=Path(file_path)
            self.logger.info(f"Loading PDF document: {file_path.name}")
            if not file_path.exists():
                raise FileNotFoundError(f"PDF document not found: {file_path}")
            loader = PyPDFLoader(str(file_path))
            document = loader.load()
            self.logger.info(f"PDF document loaded successfully with {len(document)} pages")
            return document
        except Exception as e:
            self.logger.error(f"Failed to load PDF document: {str(e)}", exc_info=True)

    def load_text(self, file_path : str | Path) -> list[Document]:
        """
        Load a text document from a file path.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            List of Document objects
        """
        try:
            file_path=Path(file_path)
            self.logger.info(f"Loading text document: {file_path.name}")
            if not file_path.exists():
                raise FileNotFoundError(f"Text document not found: {file_path}")
            loader = TextLoader(str(file_path),encoding="utf-8")
            document = loader.load()
            self.logger.info(f"Text document loaded successfully with {len(document)} pages")
            return document
        except Exception as e:
            self.logger.error(f"Failed to load text document: {str(e)}", exc_info=True)
    def load_csv(self, file_path : str | Path) -> list[Document]:
        """
        Load a CSV document from a file path.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            List of Document objects
        """
        try:
            file_path=Path(file_path)
            self.logger.info(f"Loading CSV document: {file_path.name}")
            if not file_path.exists():
                raise FileNotFoundError(f"CSV document not found: {file_path}")
            loader = CSVLoader(str(file_path),encoding="utf-8")
            document = loader.load()
            self.logger.info(f"CSV document loaded successfully with {len(document)} pages")
            return document
        except Exception as e:
            self.logger.error(f"Failed to load CSV document: {str(e)}", exc_info=True)
    def load_file(self,file_path: str | Path) -> list[Document]:
        """
        Load a document from a file path based on its extension.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of Document objects
        Raises:
            ValueError: If file extension is not supported
        """
        try:
            file_path=Path(file_path)
            self.logger.info(f"Loading document: {file_path.name}")
            extension=file_path.suffix.lower()
            if extension not in self.SUPPORTED_EXTENSIONS:
                raise ValueError(
                    f"Unsupported file extension: {extension}. "
                    f"Supported: {self.SUPPORTED_EXTENSIONS}"
                )
            loaders = {
                ".pdf": self.load_pdf,
                ".txt": self.load_text,
                ".csv": self.load_csv,
            }

            return loaders[extension](file_path)
        except Exception as e:
            self.logger.error(f"Failed to load document: {str(e)}", exc_info=True)
    def load_from_upload(self,file: BinaryIO,file_name: str) -> list[Document]:
        """
        Load a document from an uploaded file.
        
        Args:
            file: Uploaded file
            file_name: Name of the uploaded file
            
        Returns:
            List of Document objects
        """
        extension=file_name.suffix.lower()
        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {extension}")
        with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as tmp_file:
            tmp_file.write(file.read())
            tmp_file_path=Path(tmp_file.name)
            try:
                documents = self.load_file(tmp_file_path)
                for doc in documents:
                    doc.metadata["source"] = file_name
                self.logger.info(f"Document loaded successfully with {len(documents)} pages")
                return documents
            finally:
                if tmp_file_path.exists():
                    tmp_file_path.unlink(missing_ok=True)        
    def split_documents(self,documents:list[Document]) -> list[Document]:
        """
        Split documents into chunks.
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of Document objects
        """
        try:
            self.logger.info(f"Splitting documents{len(documents)} into chunks")
            chunks=self.text_splitter.split_documents(documents)
            self.logger.info(f"Documents split successfully into {len(chunks)} chunks")
            return chunks
        except Exception as e:
            self.logger.error(f"Failed to split documents: {str(e)}", exc_info=True)
    def process_file(self,file_path:str |Path)->list[Document]:
        """
        Process a document by loading and chunking it.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of chunked Document objects
        """
        documents = self.load_file(file_path)
        chunks= self.split_documents(documents)
        return chunks
    def process_upload(self,file: BinaryIO,file_name: str) -> list[Document]:
        """
        Process an uploaded document by loading and chunking it.
        
        Args:
            file: Uploaded file
            file_name: Name of the uploaded file
            
        Returns:
            List of chunked Document objects
        """
        documents = self.load_from_upload(file, file_name)
        chunks= self.split_documents(documents)
        return chunks
        
                    
            
        
