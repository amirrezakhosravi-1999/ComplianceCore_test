#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main module for CAELUS: Compliance Assessment Engine Leveraging Unified Semantics.
This orchestrates the entire compliance checking pipeline.
"""

import os
import json
import logging
import argparse
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

os.environ["TRANSFORMERS_OFFLINE"] = "1"
# Import project modules
try:
    from data_ingestion import DataIngestion
    from llm_finetuning import LLMFineTuner
    from knowledge_graph import KnowledgeGraph
    from compliance_checker import ComplianceChecker
    from report_generator import ReportGenerator
except ImportError:
    # If running from another directory, add src to path
    import sys
    sys.path.append(str(Path(__file__).resolve().parent))
    from data_ingestion import DataIngestion
    from llm_finetuning import LLMFineTuner
    from knowledge_graph import KnowledgeGraph
    from compliance_checker import ComplianceChecker
    from report_generator import ReportGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('caelus.log')
    ]
)
logger = logging.getLogger(__name__)


class CAELUSPipeline:
    """Main class for orchestrating the CAELUS compliance checking pipeline."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the CAELUS pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.data_ingestion = None
        self.fine_tuner = None
        self.knowledge_graph = None
        self.compliance_checker = None
        self.report_generator = None
        
        # Create necessary directories
        self._create_directories()
        
        logger.info("CAELUS Pipeline initialized")
        
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        default_config = {
            "data_dirs": {
                "raw_pdfs": "data/raw_pdfs",
                "processed_text": "data/processed_text",
                "semantic_units": "data/semantic_units.json",
                "design_specs": "data/design_specs",
                "fine_tuning_datasets": "data/fine_tuning_datasets"
            },
            "output_dirs": {
                "models": "models",
                "knowledge_graph": "output/knowledge_graph.json",
                "compliance_reports": "output/reports"
            },
            "model_settings": {
                "base_model":"mistralai/Mistral-7B-Instruct-v0.2", #"gpt2"
                "embedding_model":"mistralai/Mistral-7B-Instruct-v0.2",
                "fine_tuned_model": None
            },
            "pipeline_settings": {
                "run_data_ingestion": True,
                "run_fine_tuning": False,
                "run_knowledge_graph": True,
                "run_compliance_check": True,
                "top_k_regulations": 10,
                "report_format": "markdown"
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                    
                # Update default config with user config
                for section, settings in user_config.items():
                    if section in default_config:
                        if isinstance(settings, dict):
                            default_config[section].update(settings)
                        else:
                            default_config[section] = settings
                            
                logger.info(f"Loaded configuration from {config_path}")
                    
            except Exception as e:
                logger.error(f"Error loading config: {e}")
                logger.info("Using default configuration")
                
        return default_config
    
    def _create_directories(self):
        """Create necessary directories for the pipeline."""
        directories = [
            self.config["data_dirs"]["raw_pdfs"],
            self.config["data_dirs"]["processed_text"],
            self.config["data_dirs"]["design_specs"],
            self.config["data_dirs"]["fine_tuning_datasets"],
            self.config["output_dirs"]["models"],
            os.path.dirname(self.config["output_dirs"]["knowledge_graph"]),
            self.config["output_dirs"]["compliance_reports"]
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            
        logger.info("Created necessary directories")
    
    def run_data_ingestion(self) -> str:
        """Run the data ingestion phase of the pipeline."""
        logger.info("Starting data ingestion phase")
        
        self.data_ingestion = DataIngestion(
            raw_pdf_dir=self.config["data_dirs"]["raw_pdfs"],
            processed_text_dir=self.config["data_dirs"]["processed_text"],
            embedding_model_name=self.config["model_settings"]["embedding_model"]
        )
        
        # Process PDFs
        logger.info("Processing regulatory PDFs")
        processed_files = self.data_ingestion.process_all_pdfs()
        
        if not processed_files:
            logger.warning("No PDFs were processed. Please add regulatory PDFs to the raw_pdfs directory.")
            return ""
            
        # Create semantic units
        logger.info("Creating semantic units")
        all_semantic_units = []
        for text_file in processed_files:
            semantic_units = self.data_ingestion.create_semantic_units(text_file)
            if semantic_units:
                all_semantic_units.extend(semantic_units)
                
        if not all_semantic_units:
            logger.warning("No semantic units were created.")
            return ""
            
        # Generate embeddings
        logger.info("Generating embeddings for semantic units")
        all_semantic_units = self.data_ingestion.generate_embeddings(all_semantic_units)
        
        # Save semantic units
        semantic_units_path = self.config["data_dirs"]["semantic_units"]
        self.data_ingestion.save_semantic_units(all_semantic_units, semantic_units_path)
        
        logger.info(f"Data ingestion completed. Generated {len(all_semantic_units)} semantic units.")
        return semantic_units_path
    
    def run_fine_tuning(self) -> str:
        """Run the LLM fine-tuning phase of the pipeline."""
        logger.info("Starting LLM fine-tuning phase")
        
        fine_tuned_model_dir = os.path.join(self.config["output_dirs"]["models"], "fine_tuned_llm")
        
        self.fine_tuner = LLMFineTuner(
            base_model_name=self.config["model_settings"]["base_model"],
            dataset_path=os.path.join(self.config["data_dirs"]["fine_tuning_datasets"], "compliance_examples.jsonl"),
            output_dir=fine_tuned_model_dir
        )
        
        # Check if fine-tuning dataset exists
        compliance_examples_path = os.path.join(self.config["data_dirs"]["fine_tuning_datasets"], "compliance_examples.jsonl")
        relations_path = os.path.join(self.config["data_dirs"]["fine_tuning_datasets"], "relations.jsonl")
        
        if os.path.exists(compliance_examples_path) and os.path.exists(relations_path):
            logger.info("Fine-tuning datasets found")
            
            # Perform fine-tuning
            logger.info("Starting LLM fine-tuning process")
            try:
                model_path = self.fine_tuner.train(
                    epochs=3, 
                    batch_size=4, 
                    learning_rate=2e-5,
                    gradient_accumulation_steps=4
                )
                logger.info(f"Fine-tuning completed successfully. Model saved to {model_path}")
                
                # Update config with fine-tuned model path
                self.config["model_settings"]["fine_tuned_model"] = model_path
                return model_path
            except Exception as e:
                logger.error(f"Fine-tuning failed: {e}")
                logger.warning("Using base model instead")
                return ""
        else:
            logger.warning("Fine-tuning datasets not found. Skipping fine-tuning phase.")
            logger.info("Please ensure compliance_examples.jsonl and relations.jsonl exist in the fine_tuning_datasets directory")
            return ""
    
    def run_knowledge_graph(self, semantic_units_path: str) -> str:
        """
        Run the knowledge graph construction phase of the pipeline.
        
        Args:
            semantic_units_path: Path to semantic units JSON file
        """
        logger.info("Starting knowledge graph construction")
        
        # Get fine-tuned model path if available
        fine_tuned_model_path = None
        if self.config["model_settings"].get("fine_tuned_models"):
            fine_tuned_model_path = self.config["model_settings"]["fine_tuned_models"].get("relation_extraction")
        
        self.knowledge_graph = KnowledgeGraph(
            fine_tuned_model_path=fine_tuned_model_path,
            base_model_name=self.config["model_settings"]["base_model"]
        )
        
        # Load semantic units
        try:
            with open(semantic_units_path, 'r', encoding='utf-8') as f:
                semantic_units = json.load(f)
                
            # Limit units for demonstration if needed
            # semantic_units = semantic_units[:20]  # Uncomment for testing
            
            # Build knowledge graph
            logger.info(f"Building knowledge graph from {len(semantic_units)} semantic units")
            graph = self.knowledge_graph.build_graph_from_semantic_units(semantic_units)
            
            # Save graph
            knowledge_graph_path = self.config["output_dirs"]["knowledge_graph"]
            self.knowledge_graph.save_graph(knowledge_graph_path)
            
            logger.info(f"Knowledge graph construction completed with {len(graph.nodes())} nodes and {len(graph.edges())} edges")
            return knowledge_graph_path
            
        except Exception as e:
            logger.error(f"Error constructing knowledge graph: {e}")
            return ""
    
    def run_compliance_check(self, design_file_path: str) -> str:
        """
        Run the compliance checking phase of the pipeline.
        
        Args:
            design_file_path: Path to design specification file
        """
        logger.info("Starting compliance checking phase")
        
        # Get fine-tuned model path if available
        fine_tuned_model_path = None
        if self.config["model_settings"].get("fine_tuned_models"):
            fine_tuned_model_path = self.config["model_settings"]["fine_tuned_models"].get("compliance_judgment")
        
        self.compliance_checker = ComplianceChecker(
            fine_tuned_model_path=fine_tuned_model_path,
            base_model_name=self.config["model_settings"]["base_model"],
            semantic_units_path=self.config["data_dirs"]["semantic_units"]
        )
        
        # Load design specification
        try:
            with open(design_file_path, 'r', encoding='utf-8') as f:
                design_text = f.read().strip()
                
            # Get top-k relevant regulations and check compliance
            top_k = self.config["pipeline_settings"]["top_k_regulations"]
            logger.info(f"Running compliance check with top {top_k} relevant regulations")
            
            compliance_results = self.compliance_checker.batch_compliance_check(design_text, top_k=top_k)
            
            # Generate compliance report
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            design_name = Path(design_file_path).stem
            report_path = f"{self.config['output_dirs']['compliance_reports']}/compliance_report_{design_name}_{timestamp}"
            
            report = self.compliance_checker.generate_compliance_report(
                compliance_results, 
                output_path=f"{report_path}.json"
            )
            
            logger.info(f"Compliance checking completed. Report saved to {report_path}.json")
            return f"{report_path}.json"
            
        except Exception as e:
            logger.error(f"Error in compliance checking: {e}")
            return ""
    
    def generate_report(self, compliance_data_path: str) -> str:
        """
        Generate formatted report for compliance results.
        
        Args:
            compliance_data_path: Path to compliance data JSON file
        """
        logger.info("Generating formatted compliance report")
        
        self.report_generator = ReportGenerator(
            output_dir=self.config["output_dirs"]["compliance_reports"]
        )
        
        try:
            # Load compliance data
            with open(compliance_data_path, 'r', encoding='utf-8') as f:
                compliance_data = json.load(f)
                
            # Generate report based on format
            report_format = self.config["pipeline_settings"]["report_format"]
            output_path = compliance_data_path.replace('.json', f'.{report_format}')
            
            if report_format == 'html':
                report_path = self.report_generator.generate_html_report(compliance_data, output_path)
            elif report_format == 'pdf':
                report_path = self.report_generator.generate_pdf_report(compliance_data, output_path)
            elif report_format == 'excel':
                report_path = self.report_generator.generate_excel_report(compliance_data, output_path)
            elif report_format == 'markdown':
                report_path = self.report_generator.generate_markdown_report(compliance_data, output_path)
            else:
                logger.warning(f"Unsupported report format: {report_format}")
                report_path = compliance_data_path
                
            logger.info(f"Report generation completed. Report saved to {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return compliance_data_path
    
    def run_pipeline(self, design_file_path: str) -> Dict[str, str]:
        """
        Run the complete CAELUS pipeline.
        
        Args:
            design_file_path: Path to design specification file
            
        Returns:
            Dictionary with paths to generated artifacts
        """
        logger.info("Starting CAELUS pipeline")
        start_time = time.time()
        
        results = {}
        
        # Step 1: Data Ingestion
        if self.config["pipeline_settings"]["run_data_ingestion"]:
            semantic_units_path = self.run_data_ingestion()
            results["semantic_units"] = semantic_units_path
        else:
            semantic_units_path = self.config["data_dirs"]["semantic_units"]
            logger.info(f"Skipping data ingestion, using existing semantic units: {semantic_units_path}")
            results["semantic_units"] = semantic_units_path
            
        # Step 2: Fine-tuning (optional)
        if self.config["pipeline_settings"]["run_fine_tuning"]:
            fine_tuned_model_path = self.run_fine_tuning()
            results["fine_tuned_model"] = fine_tuned_model_path
            
        # Step 3: Knowledge Graph
        if self.config["pipeline_settings"]["run_knowledge_graph"]:
            knowledge_graph_path = self.run_knowledge_graph(semantic_units_path)
            results["knowledge_graph"] = knowledge_graph_path
            
        # Step 4: Compliance Check
        if self.config["pipeline_settings"]["run_compliance_check"]:
            compliance_data_path = self.run_compliance_check(design_file_path)
            results["compliance_data"] = compliance_data_path
            
            # Step 5: Report Generation (if compliance check ran)
            if compliance_data_path:
                report_path = self.generate_report(compliance_data_path)
                results["report"] = report_path
                
        elapsed_time = time.time() - start_time
        logger.info(f"CAELUS pipeline completed in {elapsed_time:.2f} seconds")
        
        return results


def main():
    """Main function to run the CAELUS pipeline."""
    parser = argparse.ArgumentParser(description='CAELUS: Compliance Assessment Engine Leveraging Unified Semantics')
    
    parser.add_argument('--design', type=str, required=True,
                        help='Path to design specification file')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to configuration file')
    parser.add_argument('--skip-ingestion', action='store_true',
                        help='Skip the data ingestion phase')
    parser.add_argument('--skip-knowledge-graph', action='store_true',
                        help='Skip the knowledge graph construction phase')
    parser.add_argument('--run-fine-tuning', action='store_true',
                        help='Run the fine-tuning phase')
    parser.add_argument('--report-format', type=str, choices=['html', 'pdf', 'excel', 'markdown'], default='markdown',
                        help='Format for the compliance report')
    parser.add_argument('--regulations', type=int, default=10,
                        help='Number of most relevant regulations to check')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = CAELUSPipeline(args.config)
    
    # Update pipeline settings from command line arguments
    pipeline.config["pipeline_settings"]["run_data_ingestion"] = not args.skip_ingestion
    pipeline.config["pipeline_settings"]["run_knowledge_graph"] = not args.skip_knowledge_graph
    pipeline.config["pipeline_settings"]["run_fine_tuning"] = args.run_fine_tuning
    pipeline.config["pipeline_settings"]["report_format"] = args.report_format
    pipeline.config["pipeline_settings"]["top_k_regulations"] = args.regulations
    
    # Run pipeline
    results = pipeline.run_pipeline(args.design)
    
    # Print results
    print("\nCAELUS Pipeline Results:")
    print("========================")
    for key, path in results.items():
        print(f"{key}: {path}")
    

if __name__ == "__main__":
    main() 