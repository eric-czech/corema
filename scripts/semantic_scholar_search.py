#!/usr/bin/env python3
# type: ignore

import requests
from pathlib import Path
import json
from typing import List, Dict, Any
import time
import tempfile
from pydantic import BaseModel, Field
import fire
import logging
from tqdm import tqdm
from corema.utils.openai_api import get_structured_output

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful research assistant that identifies papers that describe the training and evaluation of foundation models.
You should only select papers that are original research papers about specific foundation models - not surveys, reviews, or papers that just use models.
The paper must describe the actual training and evaluation of a new foundation model.

You will receive a list of papers and must return a JSON object for each paper with:
1. paper_id: The paper's ID from Semantic Scholar
2. title: The paper's title
3. is_foundation_model: Boolean indicating if it's a foundation model paper
4. model_name: The name of the foundation model (preserve casing, remove whitespace). If no explicit name is given, create an appropriate one based on the paper's focus.
5. explanation: A brief explanation of why it was included or excluded"""

USER_PROMPT = """For each of the following papers, determine if it describes the training and evaluation of a specific foundation model.
Only include papers that are original research papers about training and evaluating new foundation models.
Do not include:
- Survey papers or literature reviews
- Papers that only use or fine-tune existing models
- Papers that just propose methods without actually training a model
- Papers about smaller task-specific models

Return a JSON object for each paper containing:
- paper_id: The paper's ID
- title: The paper's title
- is_foundation_model: true/false
- model_name: The name of the foundation model (if is_foundation_model is true). Preserve casing but remove whitespace. If no explicit name is given, create an appropriate one.
- explanation: Very brief explanation of decision

For example, if a paper introduces "BERT for Chemistry", the model_name should be "BERTforChemistry".
If a paper introduces a foundation model for weather prediction but doesn't name it explicitly, use something like "WeatherFM".

Here are the papers to analyze:
{papers}"""


class PaperAnalysis(BaseModel):
    """Analysis results for a batch of papers."""

    papers: List[Dict[str, Any]] = Field(
        description="List of paper analysis results. Each result must have paper_id, title, is_foundation_model, model_name (if is_foundation_model is true), and explanation."
    )


def search_semantic_scholar(
    query: str,
    year_start: int = 2022,
    year_end: int = 2025,
    offset: int = 0,
) -> Dict:
    """
    Search Semantic Scholar using the official API.
    https://api.semanticscholar.org/api-docs/
    """
    base_url = "https://api.semanticscholar.org/graph/v1/paper/search"

    params = {
        "query": query,
        "offset": offset,
        "limit": 100,
        "year": f"{year_start}-{year_end}",
        "fields": "title,year,publicationDate,citationCount,abstract,authors,venue,citationCount,openAccessPdf,url",
        "sort": "year-desc",
        "openAccessPdf": "",
    }

    headers = {"User-Agent": "Research Script", "Accept": "application/json"}

    response = requests.get(base_url, params=params, headers=headers)
    response.raise_for_status()

    data = response.json()
    return {
        "results": data.get("data", []),
        "total": data.get("total", 0),
        "offset": data.get("offset", 0),
        "next": data.get("next", None),
    }


def save_results(papers: List[Dict], output_file: str):
    """Save raw results as JSON."""
    if not papers:
        logger.warning("No papers found matching the criteria.")
        return

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to {output_file}")


def analyze_papers(papers: List[Dict], batch_size: int = 10) -> List[Dict]:
    """
    Analyze papers to identify which ones are about training and evaluating foundation models.
    Uses LLM to analyze papers in batches.

    Args:
        papers: List of papers from Semantic Scholar
        batch_size: Number of papers to process in each LLM batch

    Returns:
        List of analysis results
    """
    # Create temp directory for incremental results
    with tempfile.TemporaryDirectory(prefix="semantic_scholar_") as temp_dir:
        temp_dir = Path(temp_dir)
        logger.info(f"Created temporary directory at {temp_dir}")

        num_batches = (len(papers) + batch_size - 1) // batch_size

        # Process papers in batches with progress bar
        with tqdm(total=len(papers), desc="Analyzing papers") as pbar:
            for i in range(0, len(papers), batch_size):
                batch = papers[i : i + batch_size]
                batch_num = i // batch_size + 1
                logger.info(
                    f"Processing batch {batch_num}/{num_batches} ({len(batch)} papers)..."
                )

                # Format papers for LLM
                papers_text = []
                for p in batch:
                    papers_text.append(
                        f"Paper ID: {p.get('paperId')}\n"
                        f"Title: {p.get('title')}\n"
                        f"Abstract: {p.get('abstract', 'No abstract available')}\n"
                        f"Venue: {p.get('venue', 'No venue available')}\n"
                    )

                # Get LLM analysis
                try:
                    response = get_structured_output(
                        system_prompt=SYSTEM_PROMPT,
                        user_prompt=USER_PROMPT.format(papers="\n\n".join(papers_text)),
                        response_model=PaperAnalysis,
                        temperature=0.1,
                    )

                    # Save batch results
                    batch_results = response.papers
                    batch_file = temp_dir / f"batch_{batch_num:03d}.json"
                    with open(batch_file, "w", encoding="utf-8") as f:
                        json.dump(batch_results, f, indent=2, ensure_ascii=False)
                    logger.info(f"Saved batch {batch_num} to {batch_file}")

                    # Log progress
                    foundation_models = sum(
                        1 for p in batch_results if p.get("is_foundation_model", False)
                    )
                    logger.info(
                        f"Found {foundation_models} foundation model papers in batch {batch_num}"
                    )

                except Exception as e:
                    logger.error(f"Error processing batch {batch_num}: {str(e)}")
                    continue
                finally:
                    # Update progress bar
                    pbar.update(len(batch))
                    pbar.set_postfix({"batch": batch_num})

                time.sleep(1)  # Be nice to the API

        # Read all batch results
        logger.info("Combining batch results...")
        all_analyses = []
        for batch_file in sorted(temp_dir.glob("batch_*.json")):
            with open(batch_file) as f:
                batch_results = json.load(f)
                all_analyses.extend(batch_results)

        return all_analyses


class SemanticScholar:
    """Commands for searching and analyzing papers from Semantic Scholar."""

    def search(
        self,
        query: str = "scientific foundation model",
        year_start: int = 2022,
        year_end: int = 2025,
        max_results: int = 300,
    ):
        """
        Search Semantic Scholar for papers and save results.

        Args:
            query: Search query string
            year_start: Start year for papers
            year_end: End year for papers
            max_results: Maximum number of results to retrieve
        """
        output_dir = Path("resources")
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / "semantic_scholar_search_results.json"

        try:
            all_papers = []
            offset = 0

            while len(all_papers) < max_results:
                logger.info(f"Fetching results (offset: {offset})...")
                result = search_semantic_scholar(
                    query=query, year_start=year_start, year_end=year_end, offset=offset
                )

                papers = result["results"]
                remaining = max_results - len(all_papers)
                papers = papers[:remaining]  # Only take what we need
                all_papers.extend(papers)

                if offset == 0:
                    logger.info(f"Total results available: {result['total']}")

                logger.info(f"Retrieved {len(papers)} papers")

                if not papers or len(papers) < 100:
                    break

                offset += 100
                time.sleep(1)  # Be nice to the API

            logger.info(f"Total papers retrieved: {len(all_papers)}")
            save_results(all_papers, output_file)

        except Exception as e:
            logger.error(f"Error occurred: {str(e)}")

    def analyze(self, batch_size: int = 10):
        """
        Analyze previously retrieved papers to identify foundation models.

        Args:
            batch_size: Number of papers to analyze in each batch
        """
        output_dir = Path("resources")
        input_file = output_dir / "semantic_scholar_search_results.json"

        if not input_file.exists():
            logger.error("No search results found. Run search command first.")
            return

        try:
            # Load search results
            with open(input_file) as f:
                papers = json.load(f)

            logger.info(f"Analyzing {len(papers)} papers...")

            # Analyze papers
            analysis_results = analyze_papers(papers, batch_size)

            # Save analysis results
            analysis_output = output_dir / "semantic_scholar_analysis_results.json"
            save_results(analysis_results, analysis_output)

            # Print summary
            foundation_models = sum(
                1 for p in analysis_results if p.get("is_foundation_model", False)
            )
            logger.info(f"Found {foundation_models} foundation model papers")

        except Exception as e:
            logger.error(f"Error occurred: {str(e)}")
            # Clean up temp directory if it exists
            temp_dir = output_dir / "temp_analysis"
            if temp_dir.exists():
                for f in temp_dir.glob("*.json"):
                    f.unlink()
                temp_dir.rmdir()

    def show_foundation_models(self):
        """Show papers that were identified as foundation models."""
        output_dir = Path("resources")
        analysis_file = output_dir / "semantic_scholar_analysis_results.json"

        if not analysis_file.exists():
            logger.error("No analysis results found. Run analyze command first.")
            return

        try:
            # Load analysis results
            with open(analysis_file) as f:
                results = json.load(f)

            # Filter for foundation models
            foundation_models = [
                p for p in results if p.get("is_foundation_model", False)
            ]

            if not foundation_models:
                logger.info("No foundation models found in the analysis results.")
                return

            # Print results in a formatted way
            print("\nFoundation Model Papers")
            print("=" * 80)

            for i, paper in enumerate(foundation_models, 1):
                print(f"\n{i}. {paper['title']}")
                print("-" * 80)
                print(f"Paper ID: {paper['paper_id']}")
                print(f"Model Name: {paper.get('model_name', 'Not specified')}")
                print(f"Explanation: {paper['explanation']}")

            print(f"\nTotal foundation models found: {len(foundation_models)}")

        except Exception as e:
            logger.error(f"Error reading analysis results: {str(e)}")

    def export_projects(self):
        """Export foundation model papers in projects.yaml format."""
        output_dir = Path("resources")
        search_file = output_dir / "semantic_scholar_search_results.json"
        analysis_file = output_dir / "semantic_scholar_analysis_results.json"

        if not search_file.exists() or not analysis_file.exists():
            logger.error(
                "Missing search or analysis results. Run search and analyze commands first."
            )
            return

        try:
            # Load both result files
            with open(search_file) as f:
                search_results = {p["paperId"]: p for p in json.load(f)}

            with open(analysis_file) as f:
                analysis_results = json.load(f)

            # Filter for foundation models and join with search results
            yaml_data = {"projects": []}
            for paper in analysis_results:
                if paper.get("is_foundation_model", False):
                    paper_id = paper["paper_id"]
                    if paper_id in search_results:
                        search_data = search_results[paper_id]

                        # Get paper URLs
                        paper_urls = []
                        if search_data.get("openAccessPdf", {}).get("url"):
                            paper_urls.append(search_data["openAccessPdf"]["url"])

                        yaml_data["projects"].append(
                            {
                                "project_name": paper.get("model_name", ""),
                                "paper_urls": paper_urls,
                                "semantic_scholar_paper_id": paper_id,
                            }
                        )

            # Save as YAML
            import yaml

            output_file = output_dir / "semantic_scholar_foundation_models.yaml"
            with open(output_file, "w", encoding="utf-8") as f:
                yaml.safe_dump(yaml_data, f, sort_keys=False, allow_unicode=True)

            logger.info(
                f"Exported {len(yaml_data['projects'])} foundation models to {output_file}"
            )

        except Exception as e:
            logger.error(f"Error exporting projects: {str(e)}")


if __name__ == "__main__":
    fire.Fire(SemanticScholar)
