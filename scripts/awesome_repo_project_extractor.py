#!/usr/bin/env python3
# type: ignore
from datetime import datetime
from pathlib import Path
import json
import pandas as pd
import xml.etree.ElementTree as ET
from pydantic import BaseModel, Field
from corema.utils.openai_api import get_structured_output
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def validate_date(year: int, month: int) -> bool:
    """Validate that year and month form a valid date"""
    try:
        if not (1900 <= year <= 2100):  # Reasonable year range
            return False
        if not (1 <= month <= 12):
            return False
        # Create date object to validate (using day=1)
        datetime(year, month, 1)
        return True
    except ValueError:
        return False


def parse_arxiv_date(identifier):
    """Extract date from arxiv identifier (YYMM.NNNNN)"""
    try:
        year_month = identifier.split(".")[0]
        year = int(year_month[:2])
        month = int(year_month[2:])
        # Assume 20xx for year
        full_year = 2000 + year

        if not validate_date(full_year, month):
            return None

        return f"{full_year}-{month:02d}"
    except (ValueError, IndexError):
        return None


def parse_biorxiv_date(identifier):
    """Extract date from biorxiv identifier (YYYY.MM.DD.NNNNNN)"""
    try:
        # Example: 10.1101/2024.10.15.618501
        # Split on '/' and take the second part which has the date
        date_part = identifier.split("/")[-1]
        parts = date_part.split(".")
        year = int(parts[0])
        month = int(parts[1])

        if not validate_date(year, month):
            return None

        return f"{year}-{month:02d}"
    except (ValueError, IndexError):
        return None


def extract_arxiv_metadata(data):
    """Extract title and abstract from arxiv XML data"""
    # Parse the XML feed
    root = ET.fromstring(data)

    # Find the entry element (contains paper metadata)
    entry = root.find("{http://www.w3.org/2005/Atom}entry")
    if entry is None:
        return None, None

    # Extract title and abstract
    title = entry.find("{http://www.w3.org/2005/Atom}title")
    abstract = entry.find("{http://www.w3.org/2005/Atom}summary")

    title_text = title.text if title is not None else None
    abstract_text = abstract.text if abstract is not None else None

    return title_text, abstract_text


def extract_biorxiv_metadata(data):
    """Extract title and abstract from biorxiv JSON data"""
    if not isinstance(data, dict) or "collection" not in data:
        return None, None

    collection = data["collection"]
    if not collection:
        return None, None

    paper_data = collection[0]
    title = paper_data.get("title")
    abstract = paper_data.get("abstract")

    return title, abstract


class ProjectName(BaseModel):
    """Model for project name generation."""

    project_name: str = Field(
        description="The generated project name using only letters and numbers"
    )
    explanation: str = Field(
        description="Brief explanation of why this name was chosen"
    )


class PaperAnalysis(BaseModel):
    """Analysis results for a paper."""

    project_name: str = Field(
        description="The generated project name using only letters and numbers"
    )
    explanation: str = Field(
        description="Brief explanation of why this name was chosen"
    )
    is_foundation_model: bool = Field(
        description="Whether this is a foundation model paper"
    )
    model_type_explanation: str = Field(
        description="Explanation of why this is or isn't a foundation model paper"
    )


SYSTEM_PROMPT = """You are a helpful research assistant that analyzes scientific papers to:
1. Determine if they describe the training and evaluation of a new foundation model
2. Extract or generate appropriate names for foundation model projects

You should only classify papers as foundation model papers if they are original research papers about specific foundation models - not surveys, reviews, or papers that just use models.
The paper must describe the actual training and evaluation of a new foundation model.

Rules for project names:
1. If the paper clearly proposes a model name in either the title or abstract (e.g. "We propose ModelName..." or "ModelName: A Foundation Model"), use that exact name
2. Remove any non-alphanumeric characters (spaces, hyphens, etc.) but preserve the original casing
3. Only if no model name is clearly proposed in either the title or abstract, create a descriptive name that reflects the domain and type
4. Only add an FM suffix if you had to create a name because none was proposed in the title or abstract
5. Never modify or add to names that are clearly proposed by the authors

Examples:
Title: "A Vision Model for Remote Sensing"
Abstract: "We propose SatVision, a foundation model that..."
-> project_name: "SatVision" (name found in abstract)

Title: "Chem-GPT: Large Language Model for Chemistry"
Abstract: "We evaluate our model on various tasks..."
-> project_name: "ChemGPT" (name found in title)

Title: "A Foundation Model for Chemical Property Prediction"
Abstract: "We train a large transformer model..."
-> project_name: "ChemistryFM" (no name proposed, created descriptive name)

Title: "FoMo-Bench: A Benchmark for Forest Monitoring"
Abstract: "We introduce FoMo-Net as a baseline model..."
-> project_name: "FoMoNet" (name found in abstract)

You must return your response as a JSON object with:
- project_name: The generated name using only letters and numbers
- explanation: Brief explanation of why this name was chosen
- is_foundation_model: Boolean indicating if it's a foundation model paper
- model_type_explanation: Brief explanation of why it is or isn't a foundation model paper"""

USER_PROMPT = """Analyze this paper to determine if it describes a new foundation model and extract or generate an appropriate project name.
Look for model names proposed in either the title or abstract.

Title: {title}
Abstract: {abstract}

Return a JSON object with:
1. The project name (using only letters and numbers)
2. Brief explanation of the name choice
3. Boolean indicating if it's a foundation model paper
4. Brief explanation of why it is or isn't a foundation model paper

Example response format:
{{
    "project_name": "SatVision",
    "explanation": "Used SatVision as it is the model name proposed in the abstract",
    "is_foundation_model": true,
    "model_type_explanation": "This paper introduces and trains a new vision foundation model for satellite imagery"
}}"""


def analyze_paper(title: str, abstract: str | None) -> PaperAnalysis:
    """Analyze a paper to determine if it's a foundation model and generate a name."""
    if abstract is None:
        abstract = "No abstract available"

    response = get_structured_output(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=USER_PROMPT.format(title=title, abstract=abstract),
        response_model=PaperAnalysis,
        temperature=0.1,
    )
    return response


def create_paper_record(url, data):
    """Create a record for a paper with metadata"""
    record = {"url": url, "identifier": data["identifier"]}

    # Extract title and abstract based on source
    if "arxiv.org" in url:
        record["publication_date"] = parse_arxiv_date(data["identifier"])
        title, abstract = extract_arxiv_metadata(data["data"])
    else:
        record["publication_date"] = parse_biorxiv_date(data["identifier"])
        title, abstract = extract_biorxiv_metadata(data["data"])

    record["title"] = title
    record["abstract"] = abstract

    # Analyze paper if we have a title
    if title:
        try:
            analysis = analyze_paper(title, abstract)
            record["project_name"] = analysis.project_name
            record["name_explanation"] = analysis.explanation
            record["is_foundation_model"] = analysis.is_foundation_model
            record["model_type_explanation"] = analysis.model_type_explanation
        except Exception as e:
            logger.error(f"Error analyzing paper: {e}")
            record["project_name"] = None
            record["name_explanation"] = None
            record["is_foundation_model"] = False
            record["model_type_explanation"] = f"Error during analysis: {str(e)}"

    return record


def main():
    # Read the papers JSON file
    papers_file = Path("resources/awesome_repo_papers.json")
    with papers_file.open() as f:
        papers = json.load(f)

    logger.info(f"Processing {len(papers)} papers from {papers_file!r}")

    # Create records for each paper
    records = []
    for paper in tqdm(papers, desc="Processing papers"):
        url = paper["url"]
        if "arxiv.org" in url or "biorxiv.org" in url:
            record = create_paper_record(url, paper)
            records.append(record)

    # Convert to DataFrame
    df = pd.DataFrame.from_records(records)

    # Save as CSV in resources
    csv_file = Path("resources") / "awesome_repo_projects.csv"
    df.to_csv(csv_file, index=False)
    logger.info(f"Saved {len(records)} paper records to {csv_file!r}")

    # Log summary statistics
    n_foundation = df["is_foundation_model"].sum()
    logger.info(
        f"Found {n_foundation} foundation model papers out of {len(records)} total papers"
    )

    # Log sample of foundation model papers
    logger.info("Sample of foundation model papers:")
    foundation_sample = df[df["is_foundation_model"]][
        ["title", "project_name", "name_explanation", "model_type_explanation"]
    ].head()
    for _, row in foundation_sample.iterrows():
        logger.info(f"\nTitle: {row['title']}")
        logger.info(f"Project Name: {row['project_name']}")
        logger.info(f"Name Explanation: {row['name_explanation']}")
        logger.info(f"Model Type: {row['model_type_explanation']}")
        logger.info("-" * 80)


if __name__ == "__main__":
    main()
