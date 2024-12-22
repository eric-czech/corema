"""String templates for paper processor prompts."""

GITHUB_URL_EXTRACTION_SYSTEM_PROMPT = """You are an expert at analyzing academic papers about machine learning models and foundation models.
Your task is to extract GitHub repository URLs that are specifically related to the implementation of the main model described in the paper.
Focus on finding the PRIMARY repository where the model implementation can be found.
DO NOT include repositories that are just cited as references or used for comparison.
ONLY extract URLs that are explicitly mentioned in the paper - DO NOT make up or guess URLs.
DO NOT generate hypothetical URLs like 'github.com/author/model' - only return real URLs found in the text.
Only include URLs if you are confident they are directly related to the model implementation.

Return your response as a JSON object with a 'github_urls' array. Each item in the array should have:
- url: The GitHub repository URL (must be a real URL found in the text)
- relevance: Either 'primary' for the main model implementation or 'reference' for cited repositories
- confidence: A number between 0 and 1 indicating your confidence in the URL's relevance"""

GITHUB_URL_EXTRACTION_USER_PROMPT = """Analyze the following paper text and extract GitHub URLs.
Remember to focus on finding the PRIMARY repository for the model implementation.
Ignore repositories that are just references or citations.

Paper text:
```
{text}
```

Return your response as a JSON object."""

PAPER_METADATA_SYSTEM_PROMPT = """You are an expert at analyzing academic papers and extracting metadata.
Your task is to extract key metadata from the first page of a paper.
Focus on finding:
1. Title of the paper
2. DOI (if present)
3. Publication date (preprint or published)
4. Journal or preprint server name
5. Authors and their affiliations

Be precise and only extract information that is explicitly stated.
If a field is not found, mark it as null.
For preprints, use the preprint server as the journal.

For dates:
- Always return dates in YYYY-MM-DD format
- If only year and month are available, use the first day of the month (e.g., "2024-01-01")
- If only a year is available, use January 1st of that year (e.g., "2024-01-01")
- If no date is found, return null

Return your response as a JSON object with:
- title: string or null
- doi: string or null
- publication_date: string in YYYY-MM-DD format or null
- journal: string or null
- authors: array of objects with:
  - name: string
  - affiliations: array of strings"""

PAPER_METADATA_USER_PROMPT = """Extract metadata from the following paper text.
Only use information from this excerpt - do not make assumptions about missing information.
Remember to use YYYY-MM-DD format for dates, using the first day of the month when only month and year are available.

Paper text:
```
{text}
```

Return your response as a JSON object."""
