# type: ignore
import pandas as pd
import requests
import re
import csv
import logging
import tempfile
from urllib.parse import urlparse, urlunparse

logger = logging.getLogger(__name__)


def fetch_readme(repo):
    """
    Attempt to fetch the README from the `main` branch first,
    then from `master`. Return the markdown text if found, else None.
    """
    # Construct possible raw URLs
    raw_url_main = f"https://raw.githubusercontent.com/{repo}/main/README.md"
    raw_url_master = f"https://raw.githubusercontent.com/{repo}/master/README.md"

    # Try main first
    response = requests.get(raw_url_main)
    if response.status_code == 200:
        return response.text

    # Fall back to master
    response = requests.get(raw_url_master)
    if response.status_code == 200:
        return response.text

    return None


def extract_markdown_links(markdown_text):
    """
    Return a list of tuples: (link_text, link_url)
    using a simple Markdown link pattern: [text](url).
    """
    pattern = r"\[([^\]]+)\]\((http[^\)]+)\)"
    # This captures:
    #   group(1): link text
    #   group(2): URL (starting with http)
    matches = re.findall(pattern, markdown_text)
    return matches


def main():
    logging.basicConfig(level=logging.INFO)

    # List of user/repo (not the entire GitHub URL) for each repository
    repos = [
        "apeterswu/Awesome-Bio-Foundation-Models",
        "robotics-survey/Awesome-Robotics-Foundation-Models",
        "Jack-bo1220/Awesome-Remote-Sensing-Foundation-Models",
        "YutingHe-list/Awesome-Foundation-Models-for-Advancing-Healthcare",
        "Jianing-Qiu/Awesome-Healthcare-Foundation-Models",
        "OmicsML/awesome-foundation-model-single-cell-papers",
        "shengchaochen82/Awesome-Foundation-Models-for-Weather-and-Climate",
        "lishenghui/awesome-fm-fl",
        "xmindflow/Awesome-Foundation-Models-in-Medical-Imaging",
        "usail-hkust/Awesome-Urban-Foundation-Models",
    ]

    # Prepare to write results to CSV
    output_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False
    ).name

    with open(output_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(["Repository", "Link Text", "Link URL"])

        for repo in repos:
            print(f"Fetching README from {repo} ...")
            readme_text = fetch_readme(repo)

            if readme_text is None:
                print(f"  Could not retrieve README for {repo}. Skipping.")
                continue

            # Extract all markdown links
            links = extract_markdown_links(readme_text)

            for link_text, link_url in links:
                # You can add additional heuristics here to filter
                # e.g., if "arxiv" in link_url or "doi" in link_url, etc.

                # Write to CSV
                writer.writerow([repo, link_text.strip(), link_url.strip()])

    logger.info(f"Extracted links to {output_file}.")
    df = pd.read_csv(output_file)

    logger.info(f"Found {len(df)} links in {output_file!r}")
    logger.info(df.head())

    logger.info("Filtering out URLs that are not papers...")
    df = df.pipe(
        lambda df: df[
            ~pd.Series([urlparse(url).netloc for url in df["Link URL"]]).str.contains(
                "|".join(
                    [
                        "zenodo.org",
                        "api.star-history.com",
                        "badge",
                        "img.shields.io",
                        "github.com",
                        "github.io",
                        "drive.google.com",
                        "huggingface.co",
                    ]
                )
            )
        ]
    )
    logger.info(f"Filtered result now has {len(df)} links")

    logger.info("Converting URLs to PDF URLs where possible...")

    def get_pdf_url(url):
        parsed_url = urlparse(url)

        if "arxiv" in parsed_url.netloc and "/abs/" in parsed_url.path:
            new_path = parsed_url.path.replace("/abs/", "/pdf/").split(".pdf")[0]
        elif "arxiv" in parsed_url.netloc and "/pdf/" in parsed_url.path:
            new_path = parsed_url.path.split(".pdf")[0]
        elif "biorxiv" in parsed_url.netloc and "/content/" in parsed_url.path:
            new_path = parsed_url.path.split(".full")[0] + ".full.pdf"
        else:
            return None

        return urlunparse(parsed_url._replace(path=new_path, query="", fragment=""))

    # Add PDF URLs to dataframe
    df["PDF URL"] = df["Link URL"].apply(get_pdf_url)
    logger.info("Data after adding PDF URLs:")
    df.info()
    logger.info(df.head())

    output_file = "resources/awesome_repo_links.csv"
    df.to_csv(output_file, index=False)
    logger.info(f"Done. Results saved to {output_file}")


if __name__ == "__main__":
    main()
