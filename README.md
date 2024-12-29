# Corema: Scientific Foundation Model Analysis

Corema (from Greek "Korema", meaning broom) is a project designed to "sweep" up and organize information about foundation models and large language models from research papers and associated resources. Like its namesake, it systematically gathers and tidies data, leveraging Large Language Models (LLMs) to intelligently extract and analyze information, creating a comprehensive knowledge base of model architectures, training methodologies, and technical specifications.

## How It Works

The pipeline operates through several key components:

1. **Data Collection**
   - Crawls academic papers from sources like arXiv, bioRxiv, and medRxiv
   - Downloads associated GitHub repositories
   - Crawls model documentation sites
   - Extracts metadata from all sources

2. **LLM-Powered Analysis**
   - Uses GPT-4 and other LLMs to:
     - Extract structured information from research papers
     - Generate standardized model summaries
     - Identify key technical specifications
     - Analyze training methodologies
     - Compare model architectures
   - Converts PDFs to machine-readable text
   - Creates visualizations of model characteristics

3. **Storage & Organization**
   - Maintains a structured local storage system
   - Tracks all artifacts through a global manifest
   - Preserves original PDFs and processed text
   - Stores LLM-extracted metadata in JSON format

## Getting Started

1. **Installation**
   ```bash
   # Clone the repository
   git clone <repository-url>
   cd corema

   # Create and activate conda environment
   conda env create -f environment.yml
   conda activate corema
   ```

2. **Basic Usage**
   ```bash
   # Run the collection pipeline
   python -m corema.cli collect projects.yaml 2>&1 | tee cli.log
   ```

## Results

Corema generates various visualizations to help understand trends and patterns in foundation model development:

### Model Evolution Timeline
![Model Timeline](docs/images/model_timeline.svg)

A chronological visualization of foundation model releases, showing relevant scientific fields and estimated model size.

### Dataset Characteristics
![Dataset Patterns](docs/images/dataset_wordclouds.svg)

Overview of common data sources, preprocessing techniques, and dataset characteristics used in model training.

### Scientific Fields
![Scientific Fields](docs/images/scientific_fields_wordcloud.svg)

Analysis of the specific scientific domains and research areas where foundation models are being applied, excluding general ML/AI terminology to highlight the unique application domains.

### Preprocessing Pipeline
![Preprocessing](docs/images/preprocessing_wordclouds.svg)

Comprehensive view of data preprocessing steps, including cleaning methods, transformation techniques, normalization approaches, and tokenization strategies used across different foundation models. This visualization reveals the common tools and systems used for preparing training data, showing the diversity of preprocessing pipelines in modern foundation model development.

### Training Infrastructure
![Compute Resources](docs/images/compute_resources_wordclouds.svg)

Analysis of computational resources and infrastructure used for model training, highlighting the scale of compute required.

### Model Architecture
![Architecture Patterns](docs/images/architecture_wordclouds.svg)

Word cloud visualization of common architectural patterns and components used across different foundation models.

### Training Stack
![Training Stack](docs/images/training_stack_wordclouds.svg)

Analysis of software frameworks, libraries, and tools commonly used in model training pipelines.
