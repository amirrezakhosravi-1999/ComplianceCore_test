#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module for data ingestion.
This includes processing regulatory documents and creating semantic units.
"""

import os
import json
import logging
import re
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataIngestion:
    """Class to handle data ingestion for regulatory documents."""
    
    def __init__(self, 
                 raw_pdf_dir: str = '../data/raw_pdfs',
                 processed_text_dir: str = '../data/processed_text',
                 embedding_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        """
        Initialize the DataIngestion class.
        
        Args:
            raw_pdf_dir: Directory containing raw PDF documents
            processed_text_dir: Directory to save processed text
            embedding_model_name: Name of embedding model
        """
        self.raw_pdf_dir = Path(raw_pdf_dir)
        self.processed_text_dir = Path(processed_text_dir)
        self.embedding_model_name = embedding_model_name
        
        # Ensure directories exist
        self.processed_text_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"DataIngestion initialized with raw_pdf_dir={raw_pdf_dir}, processed_text_dir={processed_text_dir}")
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text content from PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text content
        """
        logger.info(f"Extracting text from {pdf_path}")
        
        # For demo purposes, just return sample text if PDF processing isn't available
        sample_text = """Nuclear Safety Regulation Article 5.2: All cooling system connections must have standard thermal insulation with a minimum thickness of 50 mm.
Article 7.1: Seismic resistance with a minimum intensity of 0.35g is mandatory for all cooling system components.
Article 8.4: The emergency cooling system must be able to operate for at least 72 hours without an external power source in case of power failure.
Article 10.3: At least three independent protective layers must exist to prevent radioactive material leakage under accident conditions.
Article 12.5: The low-pressure safety injection system must have at least two independent pumps with separate power sources.
Article 15.1: The containment spray system must consist of at least three independent pumps."""
        
        logger.info(f"Extracted {len(sample_text)} characters of text from {pdf_path}")
        return sample_text
    
    def process_pdf(self, pdf_path: Path) -> Path:
        """
        Process a PDF file and save extracted text.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Path to saved text file
        """
        logger.info(f"Processing PDF: {pdf_path}")
        
        # Extract text from PDF
        extracted_text = self.extract_text_from_pdf(str(pdf_path))
        
        # Create output path
        text_filename = pdf_path.stem + ".txt"
        text_path = self.processed_text_dir / text_filename
        
        # Save extracted text
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(extracted_text)
            
        logger.info(f"Saved extracted text to {text_path}")
        return text_path
    
    def process_all_pdfs(self) -> List[Path]:
        """
        Process all PDFs in the raw PDF directory.
        
        Returns:
            List of paths to processed text files
        """
        logger.info("Processing all PDFs in directory")
        
        # Check if directory exists and contains PDFs
        if not self.raw_pdf_dir.exists():
            logger.warning(f"Raw PDF directory {self.raw_pdf_dir} does not exist")
            return []
        
        pdf_files = list(self.raw_pdf_dir.glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.raw_pdf_dir}")
            # Create sample file for demo purposes
            sample_path = self.processed_text_dir / "nuclear_safety_regulation.txt"
            with open(sample_path, 'w', encoding='utf-8') as f:
                f.write(self.extract_text_from_pdf("demo_pdf.pdf"))
            return [sample_path]
        
        # Process each PDF
        processed_files = []
        for pdf_path in pdf_files:
            try:
                text_path = self.process_pdf(pdf_path)
                processed_files.append(text_path)
            except Exception as e:
                logger.error(f"Error processing {pdf_path}: {e}")
        
        logger.info(f"Processed {len(processed_files)} PDF files")
        return processed_files
    
    def text_to_semantic_units(self, text: str) -> List[Dict[str, Any]]:
        """
        Split text into semantic units.
        
        Args:
            text: Input text
            
        Returns:
            List of semantic units
        """
        logger.info("Converting text to semantic units")
        
        # Split by article numbers or paragraphs
        units = []
        
        # اصلاح: شناسایی مقررات منفرد با الگوی بهتر
        # الگو برای تشخیص هر مقرره که با "Article" شروع می‌شود
        article_lines = text.split('\n')
        current_article = None
        
        for line in article_lines:
            line = line.strip()
            if not line:
                continue
                
            # بررسی اینکه آیا خط با Article شروع می‌شود
            article_match = re.match(r"(Nuclear Safety Regulation )?Article\s+(\d+\.\d+):\s+(.+)", line)
            if article_match:
                article_num = article_match.group(2)
                article_text = article_match.group(3)
                
                unit = {
                    "unit_id": f"article_{article_num.replace('.', '_')}",
                    "text": article_text.strip(),
                    "doc_id": "sample_regulation",
                    "section_title": f"Article {article_num}",
                    "page_number": 1,
                    "embedding": None
                }
                units.append(unit)
        
        # اگر هیچ Article یافت نشد، به روش قدیمی تقسیم کن
        if not units:
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            for i, paragraph in enumerate(paragraphs):
                unit = {
                    "unit_id": f"paragraph_{i+1}",
                    "text": paragraph,
                    "doc_id": "sample_text",
                    "section_title": f"Paragraph {i+1}",
                    "page_number": 1,
                    "embedding": None
                }
                units.append(unit)
        
        logger.info(f"Created {len(units)} semantic units")
        return units
    
    def create_semantic_units(self, text_file_path: Path) -> List[Dict[str, Any]]:
        """
        Create semantic units from a text file.
        
        Args:
            text_file_path: Path to text file
            
        Returns:
            List of semantic units
        """
        logger.info(f"Creating semantic units from {text_file_path}")
        
        # Read text file
        try:
            with open(text_file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                
            # Create semantic units
            semantic_units = self.text_to_semantic_units(text)
            
            # Add source file info
            for unit in semantic_units:
                unit["source_file"] = str(text_file_path)
            
            return semantic_units
        except Exception as e:
            logger.error(f"Error creating semantic units from {text_file_path}: {e}")
            return []
    
    def generate_embeddings(self, semantic_units: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for semantic units.
        
        Args:
            semantic_units: List of semantic units
            
        Returns:
            Semantic units with embeddings
        """
        logger.info(f"Generating embeddings for {len(semantic_units)} semantic units")
        
        try:
            # For demo purposes, generate random embeddings
            # In a real scenario, you would use the SentenceTransformer model
            for unit in semantic_units:
                # Generate a mock embedding (just for demo)
                import numpy as np
                mock_embedding = np.random.rand(384).tolist()  # typical embedding size
                unit["embedding"] = mock_embedding
            
            logger.info(f"Generated embeddings for {len(semantic_units)} semantic units")
            return semantic_units
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return semantic_units
    
    def save_semantic_units(self, semantic_units: List[Dict[str, Any]], output_path: str) -> None:
        """
        Save semantic units to a JSON file.
        
        Args:
            semantic_units: List of semantic units
            output_path: Path to save the output
        """
        logger.info(f"Saving {len(semantic_units)} semantic units to {output_path}")
        
        try:
            # Ensure output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                
            # Save semantic units
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(semantic_units, f, ensure_ascii=False, indent=2)
                
            logger.info(f"Saved semantic units to {output_path}")
        except Exception as e:
            logger.error(f"Error saving semantic units: {e}")


def main():
    """Main function for testing data ingestion."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process regulatory documents')
    parser.add_argument('--pdf_dir', type=str, default='../data/raw_pdfs',
                       help='Directory containing raw PDF documents')
    parser.add_argument('--output_dir', type=str, default='../data/processed_text',
                       help='Directory to save processed text')
    parser.add_argument('--output_units', type=str, default='../data/semantic_units.json',
                       help='Path to save semantic units')
    
    args = parser.parse_args()
    
    # Initialize data ingestion
    ingestion = DataIngestion(args.pdf_dir, args.output_dir)
    
    # Process PDFs
    processed_files = ingestion.process_all_pdfs()
    
    # Create semantic units
    all_units = []
    for file_path in processed_files:
        units = ingestion.create_semantic_units(file_path)
        all_units.extend(units)
    
    # Generate embeddings
    all_units = ingestion.generate_embeddings(all_units)
    
    # Save semantic units
    ingestion.save_semantic_units(all_units, args.output_units)
    
if __name__ == "__main__":
    main()
