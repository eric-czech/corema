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
from bs4 import BeautifulSoup
from datetime import datetime
import re

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
1. paper_id: The paper's ID from the source archive
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


def search_arxiv(
    query: str,
    year_start: int = 2022,
    year_end: int = 2025,
    max_results: int = 300,
) -> List[Dict]:
    """
    Search arXiv using their REST API.
    API docs: https://arxiv.org/help/api/user-manual
    """
    base_url = "http://export.arxiv.org/api/query"

    # Format date range for arXiv
    date_range = f"submittedDate:[{year_start}0101* TO {year_end}1231*]"

    # Construct the query
    search_query = f"{query} AND {date_range}"

    params = {
        "search_query": search_query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }

    # Make the request
    response = requests.get(base_url, params=params)
    response.raise_for_status()

    # Parse the XML response
    from xml.etree import ElementTree as ET

    root = ET.fromstring(response.content)

    # Define XML namespaces used in arXiv's API
    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }

    results = []
    for entry in root.findall("atom:entry", ns):
        # Extract paper ID from the id field (last part after '/')
        paper_id = entry.find("atom:id", ns).text.split("/")[-1]

        # Get paper details
        title = entry.find("atom:title", ns).text.strip()
        abstract = entry.find("atom:summary", ns).text.strip()
        published = entry.find("atom:published", ns).text[:10]  # Get just the date part

        # Get authors
        authors = []
        for author in entry.findall("atom:author", ns):
            name = author.find("atom:name", ns).text
            authors.append({"name": name})

        # Get PDF link
        links = entry.findall("atom:link", ns)
        pdf_url = next(
            (link.get("href") for link in links if link.get("title") == "pdf"), None
        )

        results.append(
            {
                "paper_id": paper_id,
                "title": title,
                "abstract": abstract,
                "authors": authors,
                "url": pdf_url,
                "venue": "arXiv",
                "year": int(published[:4]),
                "publicationDate": published,
            }
        )

    return results


def search_biorxiv_medrxiv(
    query: str,
    server: str,
    year_start: int = 2022,
    year_end: int = 2025,
    max_results: int = 300,
) -> List[Dict]:
    """
    Search bioRxiv or medRxiv using their web search interface.
    server should be either 'biorxiv' or 'medrxiv'
    """
    # Format dates for the URL
    today = datetime.now().strftime("%Y-%m-%d")

    # Construct search URL with proper encoding
    params = {
        "jcode": server.lower(),
        "toc_section": "New Results",
        "limit_from": f"{year_start}-01-01",
        "limit_to": today,
        "numresults": str(max_results),
        "sort": "relevance-rank",
        "format_result": "condensed",
    }

    # Encode the query and parameters properly
    encoded_query = query.replace(" ", "+")
    base_url = f"https://www.biorxiv.org/search/{encoded_query}"

    # URL encode each parameter
    param_str = "&".join(
        f"{k}={requests.utils.quote(str(v))}" for k, v in params.items()
    )
    url = f"{base_url}?{param_str}"

    logger.info(f"Searching URL: {url}")

    # Make the request
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    # Parse HTML response
    soup = BeautifulSoup(response.text, "html.parser")

    # Find all paper entries
    papers = soup.find_all("div", class_="highwire-cite")
    logger.info(f"Found {len(papers)} paper divs in HTML")

    results = []
    for paper in papers:
        try:
            # Extract paper details
            title_elem = paper.find("span", class_="highwire-cite-title")
            authors_elem = paper.find("span", class_="highwire-citation-authors")
            doi_elem = paper.find("span", class_="highwire-cite-metadata-doi")
            metadata_elem = paper.find("div", class_="highwire-cite-metadata")

            if not all([title_elem, doi_elem, metadata_elem]):
                logger.debug("Missing required elements")
                continue

            # Extract DOI and paper ID
            doi = (
                doi_elem.text.strip()
                .replace("doi: ", "")
                .replace("https://doi.org/", "")
            )
            paper_id = doi.split("/")[-1]

            # Get paper URL
            paper_url = f"https://www.{server}.org/content/{doi}"

            # Extract date from paper ID and metadata
            # Paper IDs are in format YYYY.MM.DD.NNNNNN
            # And metadata has format "bioRxiv YYYY.MM.DD.NNNNNN;"
            try:
                # First try to get from metadata text
                metadata_text = metadata_elem.text.strip()
                date_match = re.search(r"(\d{4}\.\d{2}\.\d{2})\.\d+", metadata_text)
                if date_match:
                    date_str = date_match.group(1)
                    pub_date = datetime.strptime(date_str, "%Y.%m.%d").strftime(
                        "%Y-%m-%d"
                    )
                    pub_year = int(date_str.split(".")[0])
                else:
                    # Try to get from paper ID
                    date_match = re.search(r"(\d{4}\.\d{2}\.\d{2})\.\d+", paper_id)
                    if date_match:
                        date_str = date_match.group(1)
                        pub_date = datetime.strptime(date_str, "%Y.%m.%d").strftime(
                            "%Y-%m-%d"
                        )
                        pub_year = int(date_str.split(".")[0])
                    else:
                        logger.debug(
                            f"Could not parse date from metadata or paper ID: {metadata_text}, {paper_id}"
                        )
                        continue
            except (ValueError, IndexError) as e:
                logger.debug(f"Error parsing date: {str(e)}")
                continue

            # Get authors
            authors = []
            if authors_elem:
                author_names = [name.strip() for name in authors_elem.text.split(",")]
                authors = [{"name": name} for name in author_names if name]

            # Create paper entry
            if year_start <= pub_year <= year_end:
                results.append(
                    {
                        "paper_id": paper_id,
                        "title": title_elem.text.strip(),
                        "abstract": "",  # We'd need to fetch the paper page to get the abstract
                        "authors": authors,
                        "url": paper_url,
                        "venue": server.capitalize(),
                        "year": pub_year,
                        "publicationDate": pub_date,
                    }
                )
                logger.debug(f"Added paper: {paper_id} - {pub_year}")

            if len(results) >= max_results:
                break

        except Exception as e:
            logger.warning(f"Error parsing paper entry: {str(e)}")
            continue

    return results[:max_results]


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
    """
    # Create temp directory for incremental results
    with tempfile.TemporaryDirectory(prefix="archive_search_") as temp_dir:
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
                        f"Paper ID: {p.get('paper_id')}\n"
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


class PreprintServerSearch:
    """Commands for searching and analyzing papers from arXiv, bioRxiv, and medRxiv."""

    def search(
        self,
        query: str = "scientific foundation model",
        year_start: int = 2022,
        year_end: int = 2025,
        max_results: int = 300,
    ):
        """
        Search arXiv, bioRxiv, and medRxiv for papers and save results.

        Args:
            query: Search query string
            year_start: Start year for papers
            year_end: End year for papers
            max_results: Maximum number of results to retrieve per archive
        """
        output_dir = Path("resources")
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / "preprint_server_search_results.json"

        try:
            all_papers = []

            # Search arXiv
            logger.info("Searching arXiv...")
            arxiv_papers = search_arxiv(query, year_start, year_end, max_results)
            all_papers.extend(arxiv_papers)
            logger.info(f"Found {len(arxiv_papers)} papers from arXiv")

            # Search bioRxiv
            logger.info("Searching bioRxiv...")
            biorxiv_papers = search_biorxiv_medrxiv(
                query, "biorxiv", year_start, year_end, max_results
            )
            all_papers.extend(biorxiv_papers)
            logger.info(f"Found {len(biorxiv_papers)} papers from bioRxiv")

            # Search medRxiv
            logger.info("Searching medRxiv...")
            medrxiv_papers = search_biorxiv_medrxiv(
                query, "medrxiv", year_start, year_end, max_results
            )
            all_papers.extend(medrxiv_papers)
            logger.info(f"Found {len(medrxiv_papers)} papers from medRxiv")

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
        input_file = output_dir / "preprint_server_search_results.json"

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
            analysis_output = output_dir / "preprint_server_analysis_results.json"
            save_results(analysis_results, analysis_output)

            # Print summary
            foundation_models = sum(
                1 for p in analysis_results if p.get("is_foundation_model", False)
            )
            logger.info(f"Found {foundation_models} foundation model papers")

        except Exception as e:
            logger.error(f"Error occurred: {str(e)}")

    def show_foundation_models(self):
        """Show papers that were identified as foundation models."""
        output_dir = Path("resources")
        analysis_file = output_dir / "preprint_server_analysis_results.json"

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
        search_file = output_dir / "preprint_server_search_results.json"
        analysis_file = output_dir / "preprint_server_analysis_results.json"

        if not search_file.exists() or not analysis_file.exists():
            logger.error(
                "Missing search or analysis results. Run search and analyze commands first."
            )
            return

        try:
            # Load both result files
            with open(search_file) as f:
                search_results = {p["paper_id"]: p for p in json.load(f)}

            with open(analysis_file) as f:
                analysis_results = json.load(f)

            # Filter for foundation models and join with search results
            yaml_data = {"projects": []}
            for paper in analysis_results:
                if paper.get("is_foundation_model", False):
                    paper_id = paper["paper_id"]
                    if paper_id in search_results:
                        search_data = search_results[paper_id]

                        # Extract year and month from publication date
                        pub_date = search_data.get("publicationDate", "")
                        if pub_date:
                            try:
                                pub_date_obj = datetime.strptime(pub_date, "%Y-%m-%d")
                                publication_date = pub_date_obj.strftime("%Y-%m")
                            except ValueError:
                                publication_date = None
                        else:
                            publication_date = None

                        project_entry = {
                            "project_name": paper.get("model_name", ""),
                            "paper_urls": [search_data["url"]],
                        }

                        if publication_date:
                            project_entry["publication_date"] = publication_date

                        if search_data["venue"].lower() == "arxiv":
                            project_entry["arxiv_paper_id"] = paper_id
                        elif search_data["venue"].lower() in ["biorxiv", "medrxiv"]:
                            project_entry["biorxiv_paper_id"] = paper_id

                        yaml_data["projects"].append(project_entry)

            # Save as YAML
            import yaml

            output_file = output_dir / "preprint_server_foundation_models.yaml"
            with open(output_file, "w", encoding="utf-8") as f:
                yaml.safe_dump(yaml_data, f, sort_keys=False, allow_unicode=True)

            logger.info(
                f"Exported {len(yaml_data['projects'])} foundation models to {output_file}"
            )

        except Exception as e:
            logger.error(f"Error exporting projects: {str(e)}")


if __name__ == "__main__":
    fire.Fire(PreprintServerSearch)
