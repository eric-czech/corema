# type: ignore
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
import json
from corema.utils.names import get_project_id
from corema.config import get_config, get_project_root
from wordcloud import WordCloud
from plotnine import (
    ggplot,
    aes,
    geom_bar,
    theme,
    element_text,
    labs,
    coord_flip,
    theme_minimal,
)
import fire
import logging

# Set style and backend for SVG output
plt.style.use("default")
sns.set_theme()
plt.switch_backend("svg")

# Define images directory as absolute path
IMAGES_DIR = get_project_root() / "docs" / "images"
IMAGES_DIR.mkdir(exist_ok=True, parents=True)

# Default WordCloud parameters
DEFAULT_WORDCLOUD_PARAMS = {
    "width": 1600,
    "height": 800,
    "background_color": "black",
    "min_font_size": 20,
    "max_font_size": 120,
    "contour_width": 3,
    "contour_color": "white",
    "relative_scaling": 0.0,
    "colormap": "Pastel1",  # Using Pastel1 colormap which has light, vibrant colors
}

# Set up logger
logger = logging.getLogger(__name__)


def get_publication_date(project_id, projects_data):
    """Get publication date from projects.yaml, paper metadata, or preprint URLs."""
    dates = []
    logger.debug(f"\nProcessing {project_id}:")

    # Try projects.yaml
    if project_id in projects_data:
        project = projects_data[project_id]
        if "publication_date" in project:
            try:
                date_str = project["publication_date"]
                # Handle YYYY-MM format
                if len(date_str.split("-")) == 2:
                    date = pd.to_datetime(date_str + "-01")
                    logger.debug(f"  Found date in projects.yaml: {date}")
                    dates.append(date)
                else:
                    date = pd.to_datetime(date_str)
                    logger.debug(f"  Found date in projects.yaml: {date}")
                    dates.append(date)
            except Exception as e:
                logger.debug(
                    f"  Could not parse date {date_str} from projects.yaml: {e}"
                )

    # Try paper metadata
    paper_dir = Path("data/projects") / project_id / "paper"
    if paper_dir.exists():
        json_files = list(paper_dir.glob("*.json"))
        if json_files:
            try:
                with open(json_files[0], "r") as f:
                    paper_data = json.load(f)
                    if "publication_date" in paper_data:
                        try:
                            date = pd.to_datetime(paper_data["publication_date"])
                            if pd.notna(date):
                                logger.debug(f"  Found date in paper metadata: {date}")
                                dates.append(date)
                        except Exception as e:
                            logger.debug(
                                f"  Could not parse date from paper metadata: {e}"
                            )
            except Exception as e:
                logger.debug(f"  Error reading paper metadata: {e}")

    # Try extracting from preprint URLs in projects.yaml
    if project_id in projects_data and "paper_urls" in projects_data[project_id]:
        for url in projects_data[project_id]["paper_urls"]:
            logger.debug(f"  Checking URL: {url}")
            url = url.lower()

            try:
                # Handle arXiv URLs
                if "arxiv.org" in url:
                    arxiv_id = url.split("/")[-1].split("v")[0].replace(".pdf", "")
                    logger.debug(f"    arXiv ID: {arxiv_id}")

                    # Try YYMM format
                    if len(arxiv_id) >= 4 and arxiv_id[:4].isdigit():
                        year = "20" + arxiv_id[:2]
                        month = arxiv_id[2:4]
                        if 1 <= int(month) <= 12:
                            date = pd.to_datetime(f"{year}-{month}-01")
                            logger.debug(f"    Extracted date: {date}")
                            dates.append(date)

                # Handle bioRxiv/medRxiv URLs
                elif any(x in url for x in ["biorxiv.org", "medrxiv.org"]):
                    if "10.1101/" in url:
                        date_str = url.split("10.1101/")[1].split(".")[0]
                        logger.debug(f"    bioRxiv/medRxiv date string: {date_str}")
                        if len(date_str) == 10:  # YYYY.MM.DD
                            year = date_str[:4]
                            month = date_str[5:7]
                            day = date_str[8:10]
                            date = pd.to_datetime(f"{year}-{month}-{day}")
                            logger.debug(f"    Extracted date: {date}")
                            dates.append(date)
            except Exception as e:
                logger.debug(f"    Error parsing URL: {e}")

    # Filter out None values and get earliest date
    valid_dates = [d for d in dates if pd.notna(d)]
    if valid_dates:
        earliest_date = min(valid_dates)
        logger.debug(f"  Using earliest date: {earliest_date}")
        return earliest_date

    logger.debug("  No date found")
    return None


def get_project_fields(df):
    """Get scientific fields for each project."""
    project_fields = {}
    for idx, row in df.iterrows():
        fields = row["scientific_fields"]
        if isinstance(fields, (list, np.ndarray)):
            fields = [str(f).lower().strip() for f in fields if f is not None]
            project_fields[row["project_id"]] = fields
        else:
            project_fields[row["project_id"]] = []
    return project_fields


def load_model_summary_dataset():
    """Load and prepare the model summary dataset for visualizations."""
    # Get config for data paths
    config = get_config()
    base_path = Path(config["storage"]["base_path"])
    results_path = base_path / "results" / "model_summary" / "model_summaries.jsonl"

    # Read the model summaries
    if not results_path.exists():
        raise FileNotFoundError(f"Model summaries file not found at {results_path}")

    model_summaries = pd.read_json(results_path, lines=True)

    # Read projects.yaml
    with open("projects.yaml", "r") as f:
        projects_list = yaml.safe_load(f)
        # Convert list to dict with project_id as key
        projects = {get_project_id(p["project_name"]): p for p in projects_list}

    # Add publication dates and project names
    model_summaries["publication_date"] = model_summaries["project_id"].apply(
        lambda x: get_publication_date(x, projects)
    )
    model_summaries["project_name"] = model_summaries["project_id"].apply(
        lambda x: projects[x]["project_name"] if x in projects else x
    )

    # Get project fields
    project_fields = get_project_fields(model_summaries)

    # Sort by publication date
    model_summaries = model_summaries.sort_values("publication_date")

    return model_summaries, project_fields


def create_model_table(df):
    """Create a table visualization of model IDs, names, and publication dates.

    Args:
        df: DataFrame containing model summaries
    """
    # Create figure and axis
    _, ax = plt.subplots(figsize=(12, len(df) * 0.3))
    ax.axis("tight")
    ax.axis("off")

    # Prepare data for table
    table_data = df[["project_id", "project_name", "publication_date"]].copy()
    table_data["publication_date"] = table_data["publication_date"].dt.strftime(
        "%Y-%m-%d"
    )
    table_data = table_data.fillna("Unknown")

    # Create table
    table = ax.table(
        cellText=table_data.values,
        colLabels=["Project ID", "Project Name", "Publication Date"],
        cellLoc="left",
        loc="center",
        colWidths=[0.3, 0.4, 0.3],
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    # Style header
    for j, cell in enumerate(
        table._cells[(0, j)] for j in range(len(table_data.columns))
    ):
        cell.set_text_props(weight="bold")
        cell.set_facecolor("#E6E6E6")

    # Adjust layout and save
    plt.title("Foundation Models Overview", pad=20)
    plt.savefig(IMAGES_DIR / "model_table.svg", bbox_inches="tight", format="svg")
    plt.close()


def create_timeline_visualization(df, project_fields):
    """Create a timeline visualization showing models and their scientific fields."""
    # Filter out models without dates
    df_with_dates = df[df["publication_date"].notna()].copy()

    # Create figure
    plt.figure(figsize=(14, len(df_with_dates) * 0.5))

    # Extract parameter counts from architecture dictionary
    def get_param_count(arch):
        if isinstance(arch, dict) and "parameter_count" in arch:
            counts = arch["parameter_count"]
            if isinstance(counts, (list, np.ndarray)) and len(counts) > 0:
                # Convert to numeric and get max, handling non-numeric values
                numeric_counts = pd.to_numeric(pd.Series(counts), errors="coerce")
                max_count = numeric_counts.max()
                # Ignore counts less than 1M
                if pd.notna(max_count) and max_count >= 1e6:
                    return max_count
        return None

    parameter_counts = df_with_dates["architecture"].apply(get_param_count)

    # Calculate dot sizes based on parameter counts
    # Use log scale for better visualization, add 1 to handle zeros
    # More dramatic scaling for size differences
    log_params = np.log10(parameter_counts.fillna(0) + 1)
    min_log = log_params[log_params > 0].min() if any(log_params > 0) else 0
    max_log = log_params.max()

    # Normalize to 0-1 range and apply power to create more dramatic differences
    if max_log > min_log:
        normalized_sizes = ((log_params - min_log) / (max_log - min_log)) ** 2
    else:
        normalized_sizes = log_params * 0

    # Scale to final size range - use small fixed size (25) for missing values
    sizes = np.where(
        parameter_counts.isna(), 25, 5 + normalized_sizes * 495
    )  # Range from 5 to 500 for known values

    # Create color array - blue for known parameters, grey for unknown
    colors = ["grey" if pd.isna(p) else "blue" for p in parameter_counts]

    # Plot timeline with sized dots
    y_positions = range(len(df_with_dates))
    plt.scatter(
        df_with_dates["publication_date"], y_positions, s=sizes, c=colors, alpha=0.6
    )

    # Calculate date range for offset scaling
    date_range = (
        df_with_dates["publication_date"].max()
        - df_with_dates["publication_date"].min()
    ).days
    offset = pd.Timedelta(days=date_range * 0.01)  # 1% of the date range

    # Common fields to filter out
    common_fields = {
        "machine learning",
        "deep learning",
        "artificial intelligence",
        "computer vision",
        "data science",
        "natural language processing",
        "neural networks",
        "data analysis",
        "model evaluation",
        "transformers",
        "foundation models",
        "self-supervised learning",
        "computational science",
        "information retrieval",
        "statistical analysis",
    }

    # Add model names and fields
    for idx, (_, row) in enumerate(df_with_dates.iterrows()):
        # Determine if this is the earliest model
        is_earliest = row["publication_date"] == df_with_dates["publication_date"].min()

        # Add model name with offset
        text_pos = (
            row["publication_date"] + offset
            if is_earliest
            else row["publication_date"] - offset
        )
        model_text = f"{row['project_name']}"

        # Get parameter count and format in B/M notation
        param_count = get_param_count(row["architecture"])
        if param_count is not None:
            if param_count >= 1e9:
                model_text += f" ({param_count/1e9:.0f}B)"
            else:
                model_text += f" ({param_count/1e6:.0f}M)"

        plt.text(
            text_pos,
            idx,
            f"{model_text} ",
            horizontalalignment="left" if is_earliest else "right",
            verticalalignment="bottom",
            fontsize=8,
        )

        # Filter and format fields
        fields = project_fields[row["project_id"]]
        if fields:
            # Remove common fields and convert to title case
            filtered_fields = [f.title() for f in fields if f not in common_fields][:5]
            if filtered_fields:
                field_text = ", ".join(filtered_fields)
                if len(filtered_fields) < len(
                    [f for f in fields if f not in common_fields]
                ):
                    field_text += "..."
                plt.text(
                    text_pos,
                    idx,
                    f"{field_text} ",
                    horizontalalignment="left" if is_earliest else "right",
                    verticalalignment="top",
                    fontsize=6,
                    alpha=0.7,
                )

    # Customize the plot
    plt.gca().yaxis.set_visible(False)  # Hide y-axis labels
    plt.grid(True, axis="x", alpha=0.3)
    plt.title(
        "Foundation Models Timeline\n(Dot size indicates parameter count, grey dots = unknown)"
    )

    # Format x-axis
    plt.gcf().autofmt_xdate()  # Rotate and align the tick labels

    # Add minimal padding to prevent text cutoff
    plt.margins(y=0.03)

    # Save the visualization
    plt.savefig(IMAGES_DIR / "model_timeline.svg", bbox_inches="tight", format="svg")
    plt.close()


def create_category_wordclouds(
    df, column, categories, title_prefix, output_file, n_cols=2
):
    """Create word cloud visualizations for categories in a column."""
    # Create a figure with subplots for each category
    n_rows = (len(categories) + n_cols - 1) // n_cols
    plt.figure(figsize=(10 * n_cols, 5 * n_rows))

    for idx, category in enumerate(categories, 1):
        # Create a dictionary to track which projects mention each item
        item_projects = {}

        # Collect all items and their associated project IDs
        for _, row in df.iterrows():
            project_id = row["project_id"]
            data = row[column]
            if isinstance(data, dict) and category in data:
                items = data[category]
                if isinstance(items, (list, np.ndarray)):
                    for item in items:
                        if item is not None:
                            item_str = str(item)
                            if item_str not in item_projects:
                                item_projects[item_str] = set()
                            item_projects[item_str].add(project_id)
            elif (
                isinstance(data, (list, np.ndarray)) and category == column
            ):  # Handle list data directly
                for item in data:
                    if item is not None:
                        item_str = str(item)
                        if item_str not in item_projects:
                            item_projects[item_str] = set()
                        item_projects[item_str].add(project_id)

        if item_projects:  # Only create word cloud if there are items
            # Create frequency dictionary based on number of unique projects
            freq_dict = {
                item: len(projects) for item, projects in item_projects.items()
            }

            if freq_dict:  # Only create word cloud if there are frequencies
                # Create subplot with border
                plt.subplot(n_rows, n_cols, idx)

                # Generate word cloud from frequencies
                wordcloud = WordCloud(
                    **DEFAULT_WORDCLOUD_PARAMS
                ).generate_from_frequencies(freq_dict)

                # Display word cloud
                plt.imshow(wordcloud, interpolation="bilinear")
                plt.axis("off")  # Turn off axis
                plt.box(False)  # Turn off subplot border
                plt.title(f'{title_prefix}: {category.replace("_", " ").title()}')

    plt.tight_layout(pad=3.0)
    output_path = IMAGES_DIR / output_file
    plt.savefig(output_path, bbox_inches="tight", format="svg")
    plt.close()


def create_preprocessing_wordclouds(df):
    """Create word cloud visualizations for preprocessing categories."""
    categories = [
        "tools",
        "systems",
        "data_cleaning",
        "data_transformation",
        "data_augmentation",
        "normalization",
        "tokenization",
        "other_tools",
    ]
    create_category_wordclouds(
        df=df,
        column="preprocessing",
        categories=categories,
        title_prefix="Preprocessing",
        output_file="preprocessing_wordclouds.svg",
    )


def create_dataset_wordclouds(df):
    """Create word cloud visualizations for dataset categories."""
    categories = ["modalities"]
    create_category_wordclouds(
        df=df,
        column="dataset",
        categories=categories,
        title_prefix="Dataset",
        output_file="dataset_wordclouds.svg",
        n_cols=1,
    )


def create_scientific_fields_wordcloud(df):
    """Create word cloud visualization for scientific fields."""
    # Define generic terms to filter out (case-insensitive)
    generic_terms = {
        "machine learning",
        "deep learning",
        "artificial intelligence",
        "computer vision",
        "data science",
        "natural language processing",
        "neural networks",
        "data analysis",
        "model evaluation",
        "transformers",
        "foundation models",
        "self-supervised learning",
        "computational science",
        "information retrieval",
        "statistical analysis",
        "ai",
        "ml",
        "nlp",
        "cv",
    }

    # Create a copy of the dataframe to avoid modifying the original
    df_filtered = df.copy()

    # Filter out generic terms from scientific_fields
    df_filtered["scientific_fields"] = df_filtered["scientific_fields"].apply(
        lambda fields: [
            field
            for field in (fields or [])
            if field and str(field).lower().strip() not in generic_terms
        ]
    )

    create_category_wordclouds(
        df=df_filtered,
        column="scientific_fields",
        categories=["scientific_fields"],
        title_prefix="Scientific Fields",
        output_file="scientific_fields_wordcloud.svg",
        n_cols=1,
    )


def create_training_details_wordclouds(df):
    """Create word cloud visualizations for training details categories."""
    categories = [
        "parallelization",
        "checkpointing",
        "optimization_methods",
        "regularization",
        "loss_functions",
        "training_techniques",
    ]
    create_category_wordclouds(
        df=df,
        column="training_details",
        categories=categories,
        title_prefix="Training Details",
        output_file="training_details_wordclouds.svg",
    )


def create_training_platform_wordclouds(df):
    """Create word cloud visualizations for training platform categories."""
    categories = ["cloud_provider", "hpc_system", "training_service", "other_platforms"]
    create_category_wordclouds(
        df=df,
        column="training_platform",
        categories=categories,
        title_prefix="Training Platform",
        output_file="training_platform_wordclouds.svg",
    )


def create_architecture_wordclouds(df):
    """Create word cloud visualizations for architecture categories."""
    categories = ["model_type", "architecture_type", "key_components"]
    create_category_wordclouds(
        df=df,
        column="architecture",
        categories=categories,
        title_prefix="Architecture",
        output_file="architecture_wordclouds.svg",
    )


def create_compute_resources_wordclouds(df):
    """Create word cloud visualizations for compute resources categories."""
    categories = ["training_time", "cost_estimate", "gpu_type", "other_hardware"]
    create_category_wordclouds(
        df=df,
        column="compute_resources",
        categories=categories,
        title_prefix="Compute Resources",
        output_file="compute_resources_wordclouds.svg",
    )


def load_paper_metadata():
    """Load paper metadata from all projects.

    Returns:
        pd.DataFrame: DataFrame containing paper metadata with columns:
            - project_id: str
            - project_name: str
            - paper_url: str
            - title: str
            - doi: str
            - publication_date: datetime
            - journal: str
            - authors: List[Dict] with keys 'name' and 'affiliations'
    """
    # Get config for data paths
    config = get_config()
    base_path = Path(config["storage"]["base_path"])

    # Read projects.yaml for project names
    with open("projects.yaml", "r") as f:
        projects_list = yaml.safe_load(f)
        # Convert list to dict with project_id as key
        projects = {get_project_id(p["project_name"]): p for p in projects_list}

    # Collect all paper metadata
    metadata_list = []
    for project_id, project in projects.items():
        paper_dir = base_path / "projects" / project_id / "paper"
        if not paper_dir.exists():
            continue

        # Find all JSON files in paper directory
        json_files = list(paper_dir.glob("*.json"))
        if not json_files:
            continue

        # Read each JSON file
        for json_file in json_files:
            try:
                with open(json_file, "r") as f:
                    metadata = json.load(f)

                # Add project info
                metadata["project_id"] = project_id
                metadata["project_name"] = project["project_name"]
                metadata["paper_url"] = json_file.stem  # URL hash is filename

                metadata_list.append(metadata)
            except Exception as e:
                logger.warning(f"Error reading {json_file}: {e}")

    # Convert to DataFrame
    df = pd.DataFrame(metadata_list)

    # Convert dates to datetime
    if "publication_date" in df.columns:
        df["publication_date"] = pd.to_datetime(df["publication_date"])

    # Sort by publication date
    if "publication_date" in df.columns:
        df = df.sort_values("publication_date")

    return df


def create_affiliation_visualization(df):
    """Create visualization showing the frequency of affiliations across papers.

    Args:
        df: DataFrame from load_paper_metadata()
    """
    # Load affiliation mapping
    mapping_path = Path(__file__).parent / "resources" / "affiliation_mapping.yaml"
    with open(mapping_path, "r") as f:
        affiliation_mapping = yaml.safe_load(f)

    # Create dictionary to track projects per affiliation
    affiliation_projects = {}

    # Extract all affiliations and their associated projects
    for _, row in df.iterrows():
        project_id = row["project_id"]
        for author in row["authors"]:
            if "affiliations" in author and author["affiliations"]:
                for affiliation in author["affiliations"]:
                    # Map raw affiliation to normalized name
                    normalized_affiliation = affiliation_mapping.get(
                        affiliation, "UNKNOWN_AFFILIATION"
                    )
                    if normalized_affiliation == "UNKNOWN_AFFILIATION":
                        logger.warning(
                            f"No mapping found for affiliation: {affiliation!r}"
                        )
                        continue

                    if normalized_affiliation not in affiliation_projects:
                        affiliation_projects[normalized_affiliation] = set()
                    affiliation_projects[normalized_affiliation].add(project_id)

    # Count number of distinct projects per affiliation
    affiliation_counts = {
        affiliation: len(projects)
        for affiliation, projects in affiliation_projects.items()
    }

    # Convert to DataFrame for plotting
    plot_df = pd.DataFrame(
        {"affiliation": affiliation_counts.keys(), "count": affiliation_counts.values()}
    ).sort_values("count", ascending=False)

    # Log all affiliations and their counts
    logger.debug("\nAll affiliations and their project counts:")
    with pd.option_context(
        "display.max_rows", None, "display.max_colwidth", None, "display.width", None
    ):
        logger.debug("\n" + str(plot_df))

    # Calculate figure size based on number of organizations
    # Use 0.2 inches per org for height, minimum 8 inches
    height = max(8, len(plot_df) * 0.2)

    # Create bar plot using plotnine
    plot = (
        ggplot(plot_df, aes(x="reorder(affiliation, count)", y="count"))
        + geom_bar(stat="identity", fill="#2196F3", alpha=0.7)
        + coord_flip()
        + labs(
            title="Organizations in Foundation Model Papers",
            x="Organization",
            y="Number of Projects",
        )
        + theme_minimal()
        + theme(
            figure_size=(12, height),
            plot_title=element_text(size=14, face="bold"),
            axis_text=element_text(size=10),
            axis_title=element_text(size=12),
        )
    )

    # Save plot
    plot.save(IMAGES_DIR / "affiliations.svg", dpi=300, limitsize=False)


class ModelSummaryAnalysis:
    """CLI for generating model summary visualizations."""

    def __init__(self, log_level: str = "INFO"):
        """Initialize paths and load datasets.

        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Load datasets
        self.model_summaries, self.project_fields = load_model_summary_dataset()
        self.paper_metadata = load_paper_metadata()

        logger.info(f"Loaded {len(self.model_summaries)} models")
        logger.info(
            f"Models with publication dates: {self.model_summaries['publication_date'].notna().sum()}"
        )
        logger.info(f"Loaded {len(self.paper_metadata)} papers")

    def visualize(self, allowlist: list[str] = None, blocklist: list[str] = None):
        """Generate model summary visualizations.

        Args:
            allowlist: List of visualizations to generate. If None, generate all.
                Valid values: model_table, timeline, preprocessing, dataset, scientific_fields,
                training_details, training_platform, architecture, compute_resources, affiliations
            blocklist: List of visualizations to skip. Ignored if allowlist is provided.
                Valid values: same as allowlist
        """
        all_viz = {
            "model_table": (create_model_table, [self.model_summaries]),
            "timeline": (
                create_timeline_visualization,
                [self.model_summaries, self.project_fields],
            ),
            "preprocessing": (create_preprocessing_wordclouds, [self.model_summaries]),
            "dataset": (create_dataset_wordclouds, [self.model_summaries]),
            "scientific_fields": (
                create_scientific_fields_wordcloud,
                [self.model_summaries],
            ),
            "training_details": (
                create_training_details_wordclouds,
                [self.model_summaries],
            ),
            "training_platform": (
                create_training_platform_wordclouds,
                [self.model_summaries],
            ),
            "architecture": (create_architecture_wordclouds, [self.model_summaries]),
            "compute_resources": (
                create_compute_resources_wordclouds,
                [self.model_summaries],
            ),
            "affiliations": (create_affiliation_visualization, [self.paper_metadata]),
        }

        # Determine which visualizations to create
        if allowlist:
            invalid = set(allowlist) - set(all_viz.keys())
            if invalid:
                raise ValueError(
                    f"Invalid visualization names in allowlist: {invalid!r}"
                )
            viz_to_create = {k: v for k, v in all_viz.items() if k in allowlist}
        else:
            if blocklist:
                invalid = set(blocklist) - set(all_viz.keys())
                if invalid:
                    raise ValueError(
                        f"Invalid visualization names in blocklist: {invalid!r}"
                    )
                viz_to_create = {k: v for k, v in all_viz.items() if k not in blocklist}
            else:
                viz_to_create = all_viz

        # Create visualizations
        logger.info("\nGenerating visualizations...")
        for name, (func, args) in viz_to_create.items():
            logger.info(f"Creating {name}...")
            func(*args)
            logger.info(f"- Saved to '{IMAGES_DIR / f'{name}.svg'}'")


def main():
    """Entry point for the CLI."""
    fire.Fire(ModelSummaryAnalysis)


if __name__ == "__main__":
    main()
