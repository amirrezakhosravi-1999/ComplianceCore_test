#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple executor for CAELUS project using text files instead of PDFs.
"""

import os
import sys
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('caelus_simple.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Add the current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project modules
try:
    from knowledge_graph import KnowledgeGraph
    from compliance_checker import ComplianceChecker
    from report_generator import ReportGenerator
except ImportError as e:
    logger.error(f"Failed to import modules: {e}")
    sys.exit(1)


class SimpleExecutor:
    """Simple executor class for running CAELUS with text files."""
    
    def __init__(self):
        """Initialize the executor."""
        self.semantic_units = []
        
    def process_text_file(self, file_path):
        """
        Process a text file into semantic units.
        
        Args:
            file_path: Path to text file
            
        Returns:
            List of semantic units
        """
        logger.info(f"Processing text file: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                
            # Simple semantic unit creation
            # Split by paragraphs or specific patterns
            units = []
            article_pattern = r"Article \d+:"
            section_pattern = r"\d+\.\d+"
            
            # Split by articles first
            import re
            article_splits = re.split(article_pattern, text)
            
            # First element is usually the document title, process it separately
            if article_splits and not article_splits[0].strip().startswith(article_pattern):
                header = article_splits.pop(0).strip()
                if header:
                    units.append({
                        "text": header,
                        "type": "header",
                        "source": file_path,
                        "metadata": {
                            "document_type": "regulation"
                        }
                    })
            
            # Process each article
            for i, article_text in enumerate(article_splits, 1):
                if not article_text.strip():
                    continue
                    
                article_title = f"Article {i}"
                article_content = article_text.strip()
                
                # Add the article as a unit
                units.append({
                    "text": article_title + ": " + article_content.split('\n')[0],
                    "type": "article",
                    "source": file_path,
                    "metadata": {
                        "article_number": i,
                        "document_type": "regulation"
                    }
                })
                
                # Split by sections
                section_matches = re.finditer(section_pattern, article_content)
                
                prev_pos = 0
                for match in section_matches:
                    section_start = match.start()
                    
                    # Extract section number
                    section_text = article_content[section_start:].strip()
                    section_end = section_text.find('\n')
                    if section_end != -1:
                        section = section_text[:section_end].strip()
                    else:
                        section = section_text.strip()
                        
                    # Add as a semantic unit
                    units.append({
                        "text": section,
                        "type": "section",
                        "source": file_path,
                        "metadata": {
                            "article_number": i,
                            "section_id": section.split(' ')[0],
                            "document_type": "regulation"
                        }
                    })
                    
                    prev_pos = section_start + len(section)
            
            logger.info(f"Created {len(units)} semantic units from {file_path}")
            self.semantic_units.extend(units)
            
            # Save semantic units to a file for later use
            os.makedirs('data/processed_text', exist_ok=True)
            output_path = Path('data/processed_text') / f"{Path(file_path).stem}_units.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(units, f, ensure_ascii=False, indent=2)
                
            return units
            
        except Exception as e:
            logger.error(f"Error processing text file {file_path}: {e}")
            return []
            
    def process_design_file(self, file_path):
        """
        Process a design specification file.
        
        Args:
            file_path: Path to design spec file
            
        Returns:
            Design specification text
        """
        logger.info(f"Processing design file: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                design_text = f.read()
                
            return design_text
            
        except Exception as e:
            logger.error(f"Error processing design file {file_path}: {e}")
            return ""
            
    def run_compliance_check(self, design_text):
        """
        Run compliance check for the design against regulations.
        
        Args:
            design_text: Design specification text
            
        Returns:
            Compliance results
        """
        logger.info("Running compliance check")
        
        if not self.semantic_units:
            logger.error("No semantic units available for compliance check")
            return {}
            
        # Save semantic units to a temporary file
        semantic_units_path = "data/semantic_units.json"
        with open(semantic_units_path, 'w', encoding='utf-8') as f:
            json.dump(self.semantic_units, f, ensure_ascii=False, indent=2)
            
        # Initialize the compliance checker with semantic units
        checker = ComplianceChecker(
            base_model_name=None,  # We'll use simpler methods
            fine_tuned_model_path=None,
            semantic_units_path=semantic_units_path
        )
        
        # Run compliance check
        results = checker.batch_compliance_check(design_text, top_k=10)
        
        # Generate compliance report
        report_path = "output/compliance_report.json"
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        report = checker.generate_compliance_report(results, output_path=report_path)
        
        return results
        
    def generate_html_report(self, compliance_results):
        """
        Generate an HTML report from compliance results.
        
        Args:
            compliance_results: Compliance check results
            
        Returns:
            Path to the generated report
        """
        logger.info("Generating HTML report")
        
        # Initialize the report generator
        report_generator = ReportGenerator(output_dir="output")
        
        # Generate HTML report
        report_path = "output/compliance_report.json"
        html_path = report_generator.generate_html_report(
            compliance_results, 
            output_path="output/compliance_report.html"
        )
        
        return html_path

            
def main():
    """Main function to run the simple executor."""
    logger.info("Starting CAELUS Simple Executor")
    
    # Create the executor
    executor = SimpleExecutor()
    
    # Process regulation file
    regulation_file = "data/raw_pdfs/nuclear_safety_regulation.txt"
    executor.process_text_file(regulation_file)
    
    # Process design file
    design_file = "data/design_specs/reactor_cooling_system.txt"
    design_text = executor.process_design_file(design_file)
    
    # Run compliance check
    if design_text:
        compliance_results = executor.run_compliance_check(design_text)
        
        # Generate report
        if compliance_results:
            html_report_path = executor.generate_html_report(compliance_results)
            print(f"\nCompliance check completed. Report saved to {html_report_path}")
        else:
            print("\nCompliance check failed.")
    else:
        print("\nFailed to read design specification.")
        
    logger.info("CAELUS Simple Executor completed")
    

if __name__ == "__main__":
    main() 