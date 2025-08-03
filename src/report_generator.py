#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module for generating compliance reports.
This includes reporting on design compliance with regulatory requirements.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import pandas as pd
import jinja2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ReportGenerator:
    """Class for generating compliance reports from compliance check results."""
    
    def __init__(self, 
                 template_dir: str = '../templates',
                 output_dir: str = '../output'):
        """
        Initialize the ReportGenerator class.
        
        Args:
            template_dir: Directory containing report templates
            output_dir: Directory to save generated reports
        """
        self.template_dir = Path(template_dir)
        self.output_dir = Path(output_dir)
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Jinja2 environment for HTML reports
        self._init_jinja_env()
        
        logger.info(f"ReportGenerator initialized with template_dir={template_dir}, "
                   f"output_dir={output_dir}")
        
    def _init_jinja_env(self):
        """Initialize Jinja2 environment."""
        # Create template directory if it doesn't exist
        self.template_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a default template if none exists
        default_template_path = self.template_dir / 'report_template.html'
        if not default_template_path.exists():
            default_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ report_title }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
        .header { text-align: center; margin-bottom: 30px; }
        .summary { margin: 20px 0; padding: 20px; background-color: #f9f9f9; border-radius: 5px; }
        .score { font-size: 24px; font-weight: bold; color: {{ score_color }}; }
        .issues { margin: 20px 0; }
        .issue { padding: 15px; margin: 10px 0; border-left: 4px solid; }
        .compliant { border-color: #4CAF50; }
        .partial { border-color: #FFC107; }
        .non-compliant { border-color: #F44336; }
        .charts { display: flex; justify-content: space-around; margin: 30px 0; }
        .chart-container { width: 45%; }
        table { border-collapse: collapse; width: 100%; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #f2f2f2; }
        .footer { margin-top: 40px; font-size: 0.8em; text-align: center; color: #666; }
    </style>
</head>
<body>
    <div class="header">
        <h1>{{ report_title }}</h1>
        <p>Generated on {{ generation_date }}</p>
    </div>
    
    <div class="summary">
        <h2>Compliance Summary</h2>
        <p>Overall Compliance Score: <span class="score">{{ compliance_score }}%</span></p>
        <p>Total Regulatory Requirements Checked: {{ total_checks }}</p>
        <div>
            <p>Status Distribution:</p>
            <ul>
                {% for status, count in status_counts.items() %}
                <li>{{ status }}: {{ count }}</li>
                {% endfor %}
            </ul>
        </div>
    </div>
    
    <div class="charts">
        <div class="chart-container">
            <h3>Compliance Status Distribution</h3>
            <img src="{{ pie_chart_path }}" alt="Compliance Status Distribution" width="100%">
        </div>
        <div class="chart-container">
            <h3>Compliance Score</h3>
            <img src="{{ gauge_chart_path }}" alt="Compliance Score Gauge" width="100%">
        </div>
    </div>
    
    <div class="issues">
        <h2>Compliance Issues</h2>
        {% if issues %}
        <p>{{ issues|length }} issues found requiring attention:</p>
        {% for issue in issues %}
        <div class="issue {{ issue.status|lower|replace(' ', '-') }}">
            <h3>Issue #{{ loop.index }}</h3>
            <p><strong>Status:</strong> {{ issue.status }}</p>
            <p><strong>Regulation:</strong> {{ issue.regulation }}</p>
            <p><strong>Document:</strong> {{ issue.metadata.doc_id }}</p>
            <p><strong>Section:</strong> {{ issue.metadata.section_title }}</p>
            <p><strong>Page:</strong> {{ issue.metadata.page_number }}</p>
            <p><strong>Reasoning:</strong> {{ issue.reasoning }}</p>
        </div>
        {% endfor %}
        {% else %}
        <p>No compliance issues found.</p>
        {% endif %}
    </div>
    
    <div>
        <h2>Detailed Results</h2>
        <table>
            <tr>
                <th>Regulation</th>
                <th>Status</th>
                <th>Document</th>
                <th>Section</th>
            </tr>
            {% for result in detailed_results %}
            <tr>
                <td>{{ result.regulation|truncate(100) }}</td>
                <td>{{ result.compliance_status }}</td>
                <td>{{ result.regulation_metadata.doc_id }}</td>
                <td>{{ result.regulation_metadata.section_title }}</td>
            </tr>
            {% endfor %}
        </table>
    </div>
    
    <div class="footer">
        <p>Generated by CAELUS: Compliance Assessment Engine Leveraging Unified Semantics</p>
        <p>© {{ current_year }} Nuclear Regulatory Compliance System</p>
    </div>
</body>
</html>
"""
            with open(default_template_path, 'w', encoding='utf-8') as f:
                f.write(default_template)
                
        # Initialize Jinja2 environment
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(self.template_dir),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )
    
    def generate_charts(self, compliance_data: Dict[str, Any], output_prefix: str) -> Dict[str, str]:
        """
        Generate charts for the compliance report.
        
        Args:
            compliance_data: Dictionary with compliance data
            output_prefix: Prefix for output file paths
            
        Returns:
            Dictionary with paths to generated charts
        """
        chart_paths = {}
        
        # Create charts directory if it doesn't exist
        charts_dir = self.output_dir / 'charts'
        charts_dir.mkdir(exist_ok=True)
        
        # Extract data
        status_counts = compliance_data['summary']['status_counts']
<<<<<<< HEAD
        compliance_score = compliance_data['summary']['compliance_percentage']
=======
        compliance_score = compliance_data['summary']['compliance_score']
>>>>>>> 2f5b5099fbe38f98886d9fce0926969b4913a8af
        
        # Generate pie chart for status distribution
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        
        # Filter out zero counts for better visualization
        filtered_status_counts = {k: v for k, v in status_counts.items() if v > 0}
        labels = list(filtered_status_counts.keys())
        sizes = list(filtered_status_counts.values())
        
        # Define colors for each compliance status
        colors = {
            'Compliant': 'green',
            'Partially Compliant': 'yellow',
            'Non-Compliant': 'red',
            'Not Applicable': 'gray',
            'Insufficient Information': 'lightblue',
            'Undetermined': 'lightgray'
        }
        
        colors_list = [colors.get(label, 'lightgray') for label in labels]
        
        ax1.pie(sizes, labels=labels, colors=colors_list, autopct='%1.1f%%',
                startangle=90, shadow=True)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        plt.title('Compliance Status Distribution')
        
        pie_chart_path = charts_dir / f"{output_prefix}_status_pie.png"
        plt.savefig(pie_chart_path, dpi=300, bbox_inches='tight')
        plt.close(fig1)
        chart_paths['pie_chart_path'] = str(pie_chart_path.relative_to(self.output_dir.parent))
        
        # Generate gauge chart for compliance score
        fig2, ax2 = plt.subplots(figsize=(8, 4), subplot_kw={'polar': True})
        
        # Calculate properties for gauge chart
        score_percentage = compliance_score * 100
        gauge_min, gauge_max = 0, 100
        angle = np.linspace(np.pi, 2*np.pi, 100)
        radius = 0.8
        
        # Define color gradient for score representation
        if score_percentage >= 80:
            color = 'green'
        elif score_percentage >= 60:
            color = 'yellow'
        else:
            color = 'red'
            
        # Plot gauge background
        ax2.plot(angle, [radius] * len(angle), color='lightgray', linewidth=30, alpha=0.3)
        
        # Calculate angle for score
        score_angle = np.linspace(np.pi, np.pi + (score_percentage/100) * np.pi, 100)
        ax2.plot(score_angle, [radius] * len(score_angle), color=color, linewidth=30, alpha=0.8)
        
        # Clean up plot
        ax2.set_rticks([])  # Disable radial ticks
        ax2.set_thetagrids([180, 270, 360], labels=['0%', '50%', '100%'])
        ax2.grid(False)
        
        # Add score text in center
        ax2.text(0, 0, f"{score_percentage:.1f}%", 
                ha='center', va='center', fontsize=24, fontweight='bold')
        
        plt.title('Overall Compliance Score')
        
        gauge_chart_path = charts_dir / f"{output_prefix}_score_gauge.png"
        plt.savefig(gauge_chart_path, dpi=300, bbox_inches='tight')
        plt.close(fig2)
        chart_paths['gauge_chart_path'] = str(gauge_chart_path.relative_to(self.output_dir.parent))
        
        return chart_paths
    
    def generate_html_report(self, 
                            compliance_data: Dict[str, Any], 
                            output_path: Optional[str] = None, 
                            template_name: str = 'report_template.html') -> str:
        """
        Generate HTML report from compliance data.
        
        Args:
            compliance_data: Dictionary with compliance data
            output_path: Path to save the HTML report (optional)
            template_name: Name of the template to use
            
        Returns:
            Path to the generated HTML report
        """
        logger.info(f"Generating HTML report using template: {template_name}")
        
        # Define output path if not provided
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"compliance_report_{timestamp}.html"
        else:
            output_path = Path(output_path)
            
        # Generate charts
        output_prefix = output_path.stem
        chart_paths = self.generate_charts(compliance_data, output_prefix)
        
        # Prepare template data
        template_data = {
            'report_title': 'Nuclear Regulatory Compliance Report',
            'generation_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
<<<<<<< HEAD
            'compliance_score': f"{compliance_data['summary']['compliance_percentage']:.1f}",
=======
            'compliance_score': f"{compliance_data['summary']['compliance_score'] * 100:.1f}",
>>>>>>> 2f5b5099fbe38f98886d9fce0926969b4913a8af
            'total_checks': compliance_data['summary']['total_checks'],
            'status_counts': compliance_data['summary']['status_counts'],
            'issues': compliance_data['issues'],
            'detailed_results': compliance_data['detailed_results'],
            'current_year': datetime.now().year,
<<<<<<< HEAD
            'score_color': self._get_score_color(compliance_data['summary']['compliance_percentage'])
=======
            'score_color': self._get_score_color(compliance_data['summary']['compliance_score'])
>>>>>>> 2f5b5099fbe38f98886d9fce0926969b4913a8af
        }
        
        # Add chart paths
        template_data.update(chart_paths)
        
        # Render template
        template = self.jinja_env.get_template(template_name)
        rendered_html = template.render(**template_data)
        
        # Save HTML report
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(rendered_html)
            
        logger.info(f"HTML report saved to {output_path}")
        return str(output_path)
    
    def _get_score_color(self, score: float) -> str:
        """Get color based on compliance score."""
<<<<<<< HEAD
        score_percent = score
=======
        score_percent = score * 100
>>>>>>> 2f5b5099fbe38f98886d9fce0926969b4913a8af
        if score_percent >= 90:
            return "#4CAF50"  # Green
        elif score_percent >= 75:
            return "#8BC34A"  # Light Green
        elif score_percent >= 50:
            return "#FFC107"  # Yellow
        elif score_percent >= 25:
            return "#FF9800"  # Orange
        else:
            return "#F44336"  # Red
    
    def generate_excel_report(self, 
                             compliance_data: Dict[str, Any], 
                             output_path: Optional[str] = None) -> str:
        """
        Generate Excel report from compliance data.
        
        Args:
            compliance_data: Dictionary with compliance data
            output_path: Path to save the Excel report (optional)
            
        Returns:
            Path to the generated Excel report
        """
        logger.info("Generating Excel report")
        
        # Define output path if not provided
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"compliance_report_{timestamp}.xlsx"
        else:
            output_path = Path(output_path)
            
        # Create DataFrame from detailed results
        detailed_df = []
        for result in compliance_data['detailed_results']:
            detailed_df.append({
                'Regulatory Text': result.get('regulation', '')[:1000],  # Truncate long text
                'Compliance Status': result.get('compliance_status', 'Undetermined'),
                'Document ID': result.get('regulation_metadata', {}).get('doc_id', ''),
                'Section': result.get('regulation_metadata', {}).get('section_title', ''),
                'Page': result.get('regulation_metadata', {}).get('page_number', ''),
                'Similarity Score': result.get('regulation_metadata', {}).get('similarity', 0),
                'Reasoning': result.get('reasoning', '')[:1000]  # Truncate long text
            })
            
        # Create DataFrame from issues
        issues_df = []
        for issue in compliance_data['issues']:
            issues_df.append({
                'Status': issue.get('status', ''),
                'Regulatory Text': issue.get('regulation', '')[:1000],  # Truncate long text
                'Document ID': issue.get('metadata', {}).get('doc_id', ''),
                'Section': issue.get('metadata', {}).get('section_title', ''),
                'Page': issue.get('metadata', {}).get('page_number', ''),
                'Reasoning': issue.get('reasoning', '')[:1000]  # Truncate long text
            })
            
        # Create summary data
        summary_data = {
            'Metric': [
                'Total Checks',
                'Compliance Score',
                'Compliant Count',
                'Partially Compliant Count',
                'Non-Compliant Count',
                'Not Applicable Count',
                'Insufficient Information Count',
                'Issues Count'
            ],
            'Value': [
                compliance_data['summary']['total_checks'],
                f"{compliance_data['summary']['compliance_score'] * 100:.1f}%",
                compliance_data['summary']['status_counts'].get('Compliant', 0),
                compliance_data['summary']['status_counts'].get('Partially Compliant', 0),
                compliance_data['summary']['status_counts'].get('Non-Compliant', 0),
                compliance_data['summary']['status_counts'].get('Not Applicable', 0),
                compliance_data['summary']['status_counts'].get('Insufficient Information', 0),
                len(compliance_data['issues'])
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        
        # Create Excel writer
        writer = pd.ExcelWriter(output_path, engine='xlsxwriter')
        
        # Write each DataFrame to a different worksheet
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        pd.DataFrame(issues_df).to_excel(writer, sheet_name='Issues', index=False)
        pd.DataFrame(detailed_df).to_excel(writer, sheet_name='Detailed Results', index=False)
        
        # Access workbook and worksheet objects for formatting
        workbook = writer.book
        
        # Summary sheet formatting
        summary_sheet = writer.sheets['Summary']
        summary_sheet.set_column('A:A', 30)
        summary_sheet.set_column('B:B', 15)
        
        # Issues sheet formatting
        issues_sheet = writer.sheets['Issues']
        issues_sheet.set_column('A:A', 20)
        issues_sheet.set_column('B:B', 40)
        issues_sheet.set_column('C:C', 15)
        issues_sheet.set_column('D:D', 20)
        issues_sheet.set_column('E:E', 10)
        issues_sheet.set_column('F:F', 60)
        
        # Detailed results sheet formatting
        detailed_sheet = writer.sheets['Detailed Results']
        detailed_sheet.set_column('A:A', 40)
        detailed_sheet.set_column('B:B', 20)
        detailed_sheet.set_column('C:C', 15)
        detailed_sheet.set_column('D:D', 20)
        detailed_sheet.set_column('E:E', 10)
        detailed_sheet.set_column('F:F', 15)
        detailed_sheet.set_column('G:G', 60)
        
        # Add conditional formatting for compliance status
        compliance_format = {
            'Compliant': workbook.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100'}),
            'Partially Compliant': workbook.add_format({'bg_color': '#FFEB9C', 'font_color': '#9C5700'}),
            'Non-Compliant': workbook.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'})
        }
        
        # Apply conditional formatting to detailed results
        detailed_sheet.conditional_format('B2:B1000', {
            'type': 'cell',
            'criteria': 'equal to',
            'value': '"Compliant"',
            'format': compliance_format['Compliant']
        })
        detailed_sheet.conditional_format('B2:B1000', {
            'type': 'cell',
            'criteria': 'equal to',
            'value': '"Partially Compliant"',
            'format': compliance_format['Partially Compliant']
        })
        detailed_sheet.conditional_format('B2:B1000', {
            'type': 'cell',
            'criteria': 'equal to',
            'value': '"Non-Compliant"',
            'format': compliance_format['Non-Compliant']
        })
        
        # Save the Excel file
        writer.close()
        
        logger.info(f"Excel report saved to {output_path}")
        return str(output_path)
    
    def generate_pdf_report(self, 
                           compliance_data: Dict[str, Any], 
                           output_path: Optional[str] = None) -> str:
        """
        Generate PDF report from compliance data via HTML.
        
        Args:
            compliance_data: Dictionary with compliance data
            output_path: Path to save the PDF report (optional)
            
        Returns:
            Path to the generated PDF report
        """
        logger.info("Generating PDF report")
        
        try:
            # Generate HTML report first
            html_path = None
            if output_path:
                html_path = Path(output_path).with_suffix('.html')
            
            html_output = self.generate_html_report(compliance_data, html_path)
            
            # Define PDF output path
            pdf_path = None
            if output_path:
                pdf_path = Path(output_path)
                if pdf_path.suffix.lower() != '.pdf':
                    pdf_path = pdf_path.with_suffix('.pdf')
            else:
                pdf_path = Path(html_output).with_suffix('.pdf')
                
            # Convert HTML to PDF using weasyprint
            try:
                from weasyprint import HTML
                HTML(html_output).write_pdf(pdf_path)
                logger.info(f"PDF report saved to {pdf_path}")
                
            except ImportError:
                logger.warning("WeasyPrint not installed. Using alternative PDF conversion...")
                # Alternative: use pdfkit if available
                try:
                    import pdfkit
                    pdfkit.from_file(html_output, str(pdf_path))
                    logger.info(f"PDF report saved to {pdf_path}")
                    
                except ImportError:
                    logger.error("No PDF conversion library available. Please install WeasyPrint or pdfkit.")
                    logger.info(f"HTML report available at {html_output}")
                    return html_output
                    
            return str(pdf_path)
            
        except Exception as e:
            logger.error(f"Error generating PDF report: {e}")
            return ""
    
    def generate_markdown_report(self,
                                compliance_data: Dict[str, Any],
                                output_path: Optional[str] = None) -> str:
        """
        Generate a Markdown report from compliance data.
        
        Args:
            compliance_data: Dictionary with compliance data
            output_path: Path to save the report
            
        Returns:
            Path to generated report
        """
        logger.info("Generating Markdown report")
        
        # Set output path if not provided
        if output_path is None:
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"compliance_report_{current_time}.md"
        else:
            output_path = Path(output_path)
            
        # Extract report data
        summary = compliance_data['summary']
        issues = compliance_data.get('issues', [])
        detailed_results = compliance_data.get('detailed_results', [])
        
        # Create Markdown content
        md_content = f"""# {compliance_data.get('report_title', 'Compliance Assessment Report')}

_Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}_

## Compliance Summary

<<<<<<< HEAD
**Overall Compliance Score: {summary['compliance_percentage']:.1f}%**

Total Regulatory Requirements Checked: {summary['total_requirements']}
=======
**Overall Compliance Score: {summary['compliance_score']}%**

Total Regulatory Requirements Checked: {summary['total_checks']}
>>>>>>> 2f5b5099fbe38f98886d9fce0926969b4913a8af

### Status Distribution:
"""
        
        # Add status counts
        for status, count in summary['status_counts'].items():
            md_content += f"- {status}: {count}\n"
            
        # Add issues section
        md_content += "\n## Compliance Issues\n\n"
        
        if issues:
            md_content += f"{len(issues)} issues found requiring attention:\n\n"
            
            for i, issue in enumerate(issues):
                md_content += f"### Issue #{i+1}\n"
                md_content += f"**Status:** {issue.get('status', 'Undetermined')}\n\n"
                md_content += f"**Regulation:** {issue.get('regulation', '')}\n\n"
                
                metadata = issue.get('metadata', {})
                md_content += f"**Document:** {metadata.get('doc_id', 'N/A')}\n\n"
                md_content += f"**Section:** {metadata.get('section_title', 'N/A')}\n\n"
                md_content += f"**Page:** {metadata.get('page_number', 'N/A')}\n\n"
                md_content += f"**Reasoning:** {issue.get('reasoning', '')}\n\n"
                md_content += "---\n\n"
        else:
            md_content += "No compliance issues found.\n\n"
            
        # Add detailed results
        md_content += "## Detailed Results\n\n"
        md_content += "| Regulation | Status | Document | Section |\n"
        md_content += "| --- | --- | --- | --- |\n"
        
        for result in detailed_results:
            # Truncate regulation text for better readability
            reg_text = result.get('regulation', '')
            reg_text = reg_text[:100] + "..." if len(reg_text) > 100 else reg_text
            # Escape pipe symbols in markdown table
            reg_text = reg_text.replace('|', '\\|')
            
            status = result.get('compliance_status', 'Undetermined')
            
            metadata = result.get('regulation_metadata', {})
            doc_id = metadata.get('doc_id', 'N/A')
            section = metadata.get('section_title', 'N/A')
            
            md_content += f"| {reg_text} | {status} | {doc_id} | {section} |\n"
            
        # Add footer
        md_content += "\n\n---\n"
        md_content += "\nGenerated by CAELUS: Compliance Assessment Engine Leveraging Unified Semantics\n"
        md_content += f"\n© {datetime.now().year} Nuclear Regulatory Compliance System"
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
            
        logger.info(f"Markdown report generated at {output_path}")
        return str(output_path)

    def generate_compliance_report(self,
                                  compliance_results: List[Dict[str, Any]],
                                  output_path: Optional[str] = None,
                                  report_format: str = 'markdown') -> Dict[str, Any]:
        """
        Generate a compliance report from compliance check results.
        
        Args:
            compliance_results: List of compliance check results
            output_path: Path to save the report
            report_format: Format of the report ('markdown', 'html', 'pdf', or 'excel')
            
        Returns:
            Dictionary with report data and path to generated report
        """
        logger.info("Generating compliance report")
        
        # Analyze compliance results
        total_checks = len(compliance_results)
        compliance_statuses = [r.get('compliance_status', 'Undetermined') for r in compliance_results]
        
        # Count statuses
        status_counts = {}
        for status in compliance_statuses:
            if status in status_counts:
                status_counts[status] += 1
            else:
                status_counts[status] = 1
        
        # Calculate compliance score
        compliant_count = status_counts.get('Compliant', 0)
        partially_compliant_count = status_counts.get('Partially Compliant', 0)
        
        # Give full credit for compliant, half credit for partially compliant
        if total_checks > 0:
            compliance_score = (compliant_count + (partially_compliant_count * 0.5)) / total_checks * 100
        else:
            compliance_score = 0
        
        # Round to nearest integer
        compliance_score = round(compliance_score)
        
        # Identify non-compliant issues
        issues = []
        for result in compliance_results:
            if result.get('compliance_status', '') in ['Non-Compliant', 'Partially Compliant', 'Undetermined']:
                metadata = result.get('regulation_metadata', {})
                issues.append({
                    'status': result.get('compliance_status', 'Undetermined'),
                    'regulation': result.get('regulation', ''),
                    'design': result.get('design', ''),
                    'reasoning': result.get('reasoning', ''),
                    'metadata': {
                        'doc_id': metadata.get('doc_id', 'Unknown'),
                        'section_title': metadata.get('section_title', 'Unknown'),
                        'page_number': metadata.get('page_number', 0)
                    }
                })
        
        # Create report data structure
        compliance_data = {
            'report_title': 'Design Compliance Assessment Report',
            'generation_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'summary': {
                'compliance_score': compliance_score,
                'total_checks': total_checks,
                'status_counts': status_counts,
                'status_percentages': {
                    status: (count / total_checks * 100) if total_checks > 0 else 0 
                    for status, count in status_counts.items()
                }
            },
            'issues': issues,
            'detailed_results': compliance_results,
            'score_color': self._get_score_color(compliance_score)
        }
        
        # Generate report in the requested format
        report_path = ''
        
        if report_format.lower() == 'html':
            report_path = self.generate_html_report(compliance_data, output_path)
        elif report_format.lower() == 'pdf':
            report_path = self.generate_pdf_report(compliance_data, output_path)
        elif report_format.lower() == 'excel':
            report_path = self.generate_excel_report(compliance_data, output_path)
        elif report_format.lower() == 'markdown':
            report_path = self.generate_markdown_report(compliance_data, output_path)
        else:
            # Default to Markdown format
            report_path = self.generate_markdown_report(compliance_data, output_path)
            
        compliance_data['report_path'] = report_path
        
        logger.info(f"Compliance report generated at {report_path}")
        return compliance_data


def main():
    """Main function to demonstrate the report generator."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate compliance report')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to compliance data JSON file')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to output report file')
    parser.add_argument('--format', type=str, choices=['html', 'pdf', 'excel', 'markdown'], default='html',
                        help='Output format for the report')
    args = parser.parse_args()
    
    # Load compliance data
    with open(args.input, 'r', encoding='utf-8') as f:
        compliance_data = json.load(f)
    
    # Initialize report generator
    generator = ReportGenerator()
    
    # Generate report based on format
    if args.format == 'html':
        output_path = generator.generate_html_report(compliance_data, args.output)
        print(f"HTML report generated: {output_path}")
    elif args.format == 'pdf':
        output_path = generator.generate_pdf_report(compliance_data, args.output)
        print(f"PDF report generated: {output_path}")
    elif args.format == 'excel':
        output_path = generator.generate_excel_report(compliance_data, args.output)
        print(f"Excel report generated: {output_path}")
    elif args.format == 'markdown':
        output_path = generator.generate_markdown_report(compliance_data, args.output)
        print(f"Markdown report generated: {output_path}")


if __name__ == "__main__":
    main() 