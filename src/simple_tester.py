#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple tester script to demonstrate the CAELUS compliance checking pipeline.
This demonstrates all components working together, including:
- Data processing and semantic unit creation
- Knowledge graph construction
- LLM-based compliance checking
- Markdown report generation
"""

import os
import json
import logging
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('caelus_simple_test.log')
    ]
)
logger = logging.getLogger(__name__)

# Import project modules
try:
    from data_ingestion import DataIngestion
    from knowledge_graph import KnowledgeGraph
    from compliance_checker import ComplianceChecker
    from report_generator import ReportGenerator
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).resolve().parent))
    from data_ingestion import DataIngestion
    from knowledge_graph import KnowledgeGraph 
    from compliance_checker import ComplianceChecker
    from report_generator import ReportGenerator


def setup_data_and_directories():
    """Setup sample data and necessary directories."""
    root_dir = Path(__file__).resolve().parent.parent
    data_dir = root_dir / 'data'
    output_dir = root_dir / 'output'
    
    # Create directories
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(data_dir / 'processed_text', exist_ok=True)
    
    # Sample regulatory text
    sample_text = """Nuclear Safety Regulation Article 5.2: All cooling system connections must have standard thermal insulation with a minimum thickness of 50 mm.
Article 7.1: Seismic resistance with a minimum intensity of 0.35g is mandatory for all cooling system components.
Article 8.4: The emergency cooling system must be able to operate for at least 72 hours without an external power source in case of power failure.
Article 10.3: At least three independent protective layers must exist to prevent radioactive material leakage under accident conditions.
Article 12.5: The low-pressure safety injection system must have at least two independent pumps with separate power sources.
Article 15.1: The containment spray system must consist of at least three independent pumps."""

    # Sample design specification
    sample_design = """Cooling System Design Specification:
The cooling system connections are insulated with standard industrial-grade thermal insulation. Average thickness is 45mm.
All cooling system components are designed to withstand seismic events up to 0.25g intensity.
Emergency Cooling System - Capable of operating without external power for 96 hours.
Containment spray system consists of two independent pumps with separate power supplies."""

    # Save sample files
    sample_text_path = data_dir / 'processed_text' / 'sample_regulations.txt'
    with open(sample_text_path, 'w', encoding='utf-8') as f:
        f.write(sample_text)
    
    sample_design_path = data_dir / 'processed_text' / 'sample_design.txt'
    with open(sample_design_path, 'w', encoding='utf-8') as f:
        f.write(sample_design)
    
    logger.info(f"Sample data saved to {data_dir / 'processed_text'}")
    return sample_text_path, sample_design_path, root_dir


def process_regulations(text_file_path, root_dir):
    """Process regulations and create semantic units."""
    logger.info("Processing regulations text")
    
    # Initialize data ingestion
    data_ingestion = DataIngestion(
        processed_text_dir=str(root_dir / 'data' / 'processed_text'),
        embedding_model_name='sentence-transformers/all-MiniLM-L6-v2'
    )
    
    # Create semantic units
    semantic_units = data_ingestion.create_semantic_units(text_file_path)
    logger.info(f"Created {len(semantic_units)} semantic units")
    
    # Generate embeddings
    semantic_units_with_embeddings = data_ingestion.generate_embeddings(semantic_units)
    
    # Save semantic units
    semantic_units_path = root_dir / 'data' / 'semantic_units.json'
    data_ingestion.save_semantic_units(semantic_units_with_embeddings, str(semantic_units_path))
    logger.info(f"Saved semantic units to {semantic_units_path}")
    
    return semantic_units_path


def build_knowledge_graph(semantic_units_path, root_dir):
    """Build knowledge graph from semantic units."""
    logger.info("Building knowledge graph")
    
    # Initialize knowledge graph
    kg = KnowledgeGraph(
        base_model_name='mistralai/Mistral-7B-Instruct-v0.2',
        fine_tuned_model_path=None  # Using the base model since fine-tuned model might not be available
    )
    
    # Load semantic units
    with open(semantic_units_path, 'r', encoding='utf-8') as f:
        units = json.load(f)
    
    try:
        # Try to build the graph
        graph = kg.build_graph_from_semantic_units(units)
        
        # Save the graph
        kg_path = root_dir / 'output' / 'knowledge_graph.json'
        os.makedirs(os.path.dirname(kg_path), exist_ok=True)
        kg.save_graph(str(kg_path))
        logger.info(f"Saved knowledge graph to {kg_path}")
        
        return kg_path
    except Exception as e:
        logger.error(f"Could not build knowledge graph: {e}")
        logger.info("Continuing without knowledge graph")
        return None


def check_compliance(semantic_units_path, design_file_path, root_dir):
    """Check compliance of design against regulations."""
    logger.info("Checking compliance")
    
    # Read design specification
    with open(design_file_path, 'r', encoding='utf-8') as f:
        design_text = f.read()
    
    # Initialize compliance checker
    checker = ComplianceChecker(
        fine_tuned_model_path=None,  # Using the base model since fine-tuned model might not be available
        semantic_units_path=str(semantic_units_path)
    )
    
    # Find relevant regulations - افزایش تعداد مقررات برای بررسی بیشتر
    relevant_regulations = checker.find_relevant_regulations(design_text, top_k=10)
    logger.info(f"Found {len(relevant_regulations)} relevant regulations")
    
    # Check compliance for each relevant regulation با استفاده مستقیم از design_text
    compliance_results = []
    
    for reg in relevant_regulations:
        logger.info(f"Checking compliance for regulation: {reg['text'][:50]}...")
        result = checker.check_compliance(design_text, reg['text'])
        
        # Add metadata from the regulation
        result['regulation_metadata'] = {
            'unit_id': reg.get('unit_id', ''),
            'doc_id': reg.get('doc_id', ''),
            'section_title': reg.get('section_title', ''),
            'page_number': reg.get('page_number', 0),
            'similarity': reg.get('similarity', 0)
        }
        
        compliance_results.append(result)
    
    logger.info(f"Completed {len(compliance_results)} compliance checks")
    
    # Save compliance results
    compliance_path = root_dir / 'output' / 'compliance_results.json'
    with open(compliance_path, 'w', encoding='utf-8') as f:
        json.dump(compliance_results, f, indent=2)
    logger.info(f"Saved compliance results to {compliance_path}")
    
    return compliance_path


def generate_report(compliance_path, root_dir, report_format='markdown'):
    """Generate a formatted compliance report."""
    logger.info(f"Generating {report_format.upper()} report")
    
    # Initialize report generator
    report_generator = ReportGenerator(
        output_dir=str(root_dir / 'output')
    )
    
    # Load compliance results
    with open(compliance_path, 'r', encoding='utf-8') as f:
        compliance_results = json.load(f)
    
    # Generate report
    report_data = report_generator.generate_compliance_report(
        compliance_results,
        output_path=str(root_dir / 'output' / f'compliance_report.{report_format}'),
        report_format=report_format
    )
    
    logger.info(f"Report generated: {report_data['report_path']}")
    print(f"\nCompliance Summary:")
    print(f"Overall Compliance Score: {report_data['summary']['compliance_score']}%")
    print(f"Total Regulatory Requirements Checked: {report_data['summary']['total_checks']}")
    print("\nStatus Distribution:")
    for status, count in report_data['summary']['status_counts'].items():
        print(f"  {status}: {count}")
    
    return report_data['report_path']


def main():
    parser = argparse.ArgumentParser(description='CAELUS Simple Tester')
    parser.add_argument('--report-format', type=str, choices=['html', 'pdf', 'excel', 'markdown'], default='markdown',
                        help='Format for the compliance report')
    parser.add_argument('--skip-kg', action='store_true',
                        help='Skip knowledge graph generation (for resource-constrained environments)')
    args = parser.parse_args()
    
    print("\nCAELUS Compliance Assessment Tester")
    print("===================================\n")
    
    try:
        # Setup sample data and directories
        sample_text_path, sample_design_path, root_dir = setup_data_and_directories()
        
        # Process regulations
        semantic_units_path = process_regulations(sample_text_path, root_dir)
        
        # Build knowledge graph (if not skipped)
        if not args.skip_kg:
            kg_path = build_knowledge_graph(semantic_units_path, root_dir)
        
        # Check compliance
        compliance_path = check_compliance(semantic_units_path, sample_design_path, root_dir)
        
        # Generate report
        report_path = generate_report(compliance_path, root_dir, args.report_format)
        
        print(f"\nCompliance testing completed successfully!")
        print(f"Report saved to: {report_path}")
        
    except Exception as e:
        logger.error(f"Error in compliance testing: {e}", exc_info=True)
        print(f"\nError occurred: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 