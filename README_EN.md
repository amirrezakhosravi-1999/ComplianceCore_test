# CAELUS - AI-Powered Nuclear Regulatory Compliance Checker

**Compliance Assessment Engine Leveraging Unified Semantics**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 🚀 Overview

CAELUS is an intelligent system for assessing compliance of nuclear engineering designs against regulatory requirements and industry standards. By leveraging advanced Large Language Models (LLMs), Knowledge Graphs, and Natural Language Processing techniques, CAELUS significantly improves the accuracy and efficiency of compliance checking processes.

### 🎯 Problem Statement

Traditional compliance checking in nuclear engineering is:
- **Time-consuming**: Manual review can take weeks or months
- **Error-prone**: Human oversight of complex regulations
- **Inconsistent**: Different reviewers may reach different conclusions
- **Costly**: Requires specialized domain experts

### 💡 Solution

CAELUS automates compliance checking using:
- **Fine-tuned LLMs** for domain-specific understanding
- **Knowledge Graphs** for complex regulatory relationships
- **Semantic Search** for precise document matching
- **Automated Report Generation** with detailed explanations

## 🏗️ Architecture

The CAELUS system consists of four main components:

1. **Data Ingestion**: Extract and process regulatory texts into semantic units
2. **Knowledge Graph**: Build relationship graphs between regulatory sections
3. **Compliance Checker**: Use fine-tuned LLMs for intelligent design-regulation matching
4. **Report Generator**: Produce comprehensive compliance reports

## ✨ Key Features

- **LLM-based Compliance Detection**: Uses advanced language models instead of rule-based approaches
- **Model Fine-tuning**: Custom training on regulatory data for improved accuracy
- **Knowledge Graph Integration**: Visual representation of regulatory relationships
- **Multi-format Reports**: Generate reports in Markdown, HTML, PDF, and Excel formats
- **Semantic Unit Processing**: Intelligent text segmentation for better understanding

## 🛠️ Technical Stack

- **Core ML**: PyTorch, Transformers, Sentence-Transformers
- **LLM Fine-tuning**: LoRA/PEFT, BitsAndBytes, TRL
- **Knowledge Graph**: NetworkX, Matplotlib
- **NLP**: NLTK, SpaCy
- **Reporting**: Jinja2, WeasyPrint, XlsxWriter
- **Web Interface**: Streamlit, Plotly

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for LLM fine-tuning)
- 16GB+ RAM (32GB recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/caelus-compliance.git
cd caelus-compliance
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

## 🚀 Quick Start

### Simple Demo Run

For a quick demonstration with sample data:

```bash
python src/simple_tester.py --skip-kg
```

The `--skip-kg` flag skips the knowledge graph creation step which requires significant computational resources.

### Full Pipeline Run

```bash
python src/main.py --design data/design_specs/reactor_cooling_system.txt --report-format markdown
```

### Jupyter Notebook Demo

Explore the interactive demo:

```bash
jupyter notebook demo_notebook.ipynb
```

## 📁 Project Structure

```
caelus_compliance_project/
├── data/                          # Data directory
│   ├── raw_pdfs/                  # Original regulatory PDF files
│   ├── processed_text/            # Extracted text from PDFs
│   ├── design_specs/              # Design specification files
│   ├── fine_tuning_datasets/      # Training data for fine-tuning
│   └── semantic_units.json        # Processed semantic units
├── models/                        # Model storage
│   ├── fine_tuned_llm/           # Fine-tuned model adapters
│   └── embeddings/               # Pre-computed embeddings
├── src/                          # Source code
│   ├── data_ingestion.py         # Data processing and extraction
│   ├── knowledge_graph.py        # Knowledge graph construction
│   ├── compliance_checker.py     # Core compliance logic
│   ├── report_generator.py       # Report generation
│   ├── llm_finetuning.py        # LLM fine-tuning utilities
│   └── main.py                   # Main pipeline orchestrator
├── demo_notebook.ipynb           # Interactive demonstration
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

## 🔧 Configuration

### Command Line Arguments

- `--design`: Path to design specification file
- `--report-format`: Output format (markdown, html, pdf, excel)
- `--skip-ingestion`: Skip data processing step
- `--skip-knowledge-graph`: Skip knowledge graph creation
- `--run-fine-tuning`: Run LLM fine-tuning process
- `--regulations`: Number of relevant regulations to check

### Environment Variables

Set these environment variables for optimal performance:

```bash
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/path/to/huggingface/cache
```

## 🧠 Model Fine-tuning

CAELUS supports fine-tuning LLMs for improved performance:

1. Prepare training data in JSONL format
2. Run fine-tuning:

```bash
python src/llm_finetuning.py --dataset data/fine_tuning_datasets/compliance_examples.jsonl
```

3. The fine-tuned model will be saved in `models/fine_tuned_llm/`

## 📊 Performance Metrics

- **Processing Speed**: ~6 semantic units in 30 seconds
- **Memory Usage**: 8-16GB RAM (with GPU acceleration)
- **Accuracy**: 60-70% compliance detection accuracy (improving with domain-specific training)
- **Report Generation**: Under 1 minute for typical documents
- **Current Status**: Rule-based fallback system operational, LLM fine-tuning in progress

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Parscoders Team** for development and research
- **Nuclear Regulatory Agencies** for providing public documentation
- **Hugging Face** for transformer models and tools
- **Open Source Community** for various libraries and tools

## 📞 Support

For questions or support:
- Create an issue on GitHub
- Contact: [your-email@domain.com]
- Documentation: [Link to detailed docs]

## 🗺️ Roadmap

- [ ] Support for additional regulatory frameworks
- [ ] Multi-language support
- [ ] Enhanced web interface
- [ ] Real-time compliance monitoring
- [ ] Integration with CAD tools
- [ ] Cloud deployment options

---

 