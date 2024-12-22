# type: ignore
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
import json
from corema.utils.names import get_project_id
from wordcloud import WordCloud
from corema.config import get_config

# Set style and backend for SVG output
plt.style.use("default")
sns.set_theme()
plt.switch_backend("svg")

# Create visualizations directory if it doesn't exist
Path("docs/images").mkdir(exist_ok=True, parents=True)

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


def get_publication_date(project_id, projects_data):
    """Get publication date from projects.yaml, paper metadata, or preprint URLs."""
    dates = []
    print(f"\nProcessing {project_id}:")

    # Try projects.yaml
    if project_id in projects_data:
        project = projects_data[project_id]
        if "publication_date" in project:
            try:
                date_str = project["publication_date"]
                # Handle YYYY-MM format
                if len(date_str.split("-")) == 2:
                    date = pd.to_datetime(date_str + "-01")
                    print(f"  Found date in projects.yaml: {date}")
                    dates.append(date)
                else:
                    date = pd.to_datetime(date_str)
                    print(f"  Found date in projects.yaml: {date}")
                    dates.append(date)
            except Exception as e:
                print(f"  Could not parse date {date_str} from projects.yaml: {e}")

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
                                print(f"  Found date in paper metadata: {date}")
                                dates.append(date)
                        except Exception as e:
                            print(f"  Could not parse date from paper metadata: {e}")
            except Exception as e:
                print(f"  Error reading paper metadata: {e}")

    # Try extracting from preprint URLs in projects.yaml
    if project_id in projects_data and "paper_urls" in projects_data[project_id]:
        for url in projects_data[project_id]["paper_urls"]:
            print(f"  Checking URL: {url}")
            url = url.lower()

            try:
                # Handle arXiv URLs
                if "arxiv.org" in url:
                    arxiv_id = url.split("/")[-1].split("v")[0].replace(".pdf", "")
                    print(f"    arXiv ID: {arxiv_id}")

                    # Try YYMM format
                    if len(arxiv_id) >= 4 and arxiv_id[:4].isdigit():
                        year = "20" + arxiv_id[:2]
                        month = arxiv_id[2:4]
                        if 1 <= int(month) <= 12:
                            date = pd.to_datetime(f"{year}-{month}-01")
                            print(f"    Extracted date: {date}")
                            dates.append(date)

                # Handle bioRxiv/medRxiv URLs
                elif any(x in url for x in ["biorxiv.org", "medrxiv.org"]):
                    if "10.1101/" in url:
                        date_str = url.split("10.1101/")[1].split(".")[0]
                        print(f"    bioRxiv/medRxiv date string: {date_str}")
                        if len(date_str) == 10:  # YYYY.MM.DD
                            year = date_str[:4]
                            month = date_str[5:7]
                            day = date_str[8:10]
                            date = pd.to_datetime(f"{year}-{month}-{day}")
                            print(f"    Extracted date: {date}")
                            dates.append(date)
            except Exception as e:
                print(f"    Error parsing URL: {e}")

    # Filter out None values and get earliest date
    valid_dates = [d for d in dates if pd.notna(d)]
    if valid_dates:
        earliest_date = min(valid_dates)
        print(f"  Using earliest date: {earliest_date}")
        return earliest_date

    print("  No date found")
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


def load_standard_dataset():
    """Load and prepare the standard dataset for visualizations."""
    # Get config for data paths
    config = get_config()
    base_path = Path(config["storage"]["base_path"])
    results_path = base_path / "results" / "model_summary" / "model_summaries.jsonl"

    # Read the model summaries
    if not results_path.exists():
        raise FileNotFoundError(f"Model summaries file not found at {results_path}")

    df = pd.read_json(results_path, lines=True)

    # Read projects.yaml
    with open("projects.yaml", "r") as f:
        projects_list = yaml.safe_load(f)
        # Convert list to dict with project_id as key
        projects = {get_project_id(p["project_name"]): p for p in projects_list}

    # Add publication dates and project names
    df["publication_date"] = df["project_id"].apply(
        lambda x: get_publication_date(x, projects)
    )
    df["project_name"] = df["project_id"].apply(
        lambda x: projects[x]["project_name"] if x in projects else x
    )

    # Get project fields
    project_fields = get_project_fields(df)

    # Sort by publication date
    df = df.sort_values("publication_date")

    return df, project_fields


def create_model_table():
    """Create a table visualization of model IDs, names, and publication dates."""
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
    plt.savefig("docs/images/model_table.svg", bbox_inches="tight", format="svg")
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
    plt.savefig("docs/images/model_timeline.svg", bbox_inches="tight", format="svg")
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
    output_path = Path("docs/images") / output_file
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


# Load the standard dataset
df, project_fields = load_standard_dataset()

print(f"Loaded {len(df)} models")
print(f"Models with publication dates: {df['publication_date'].notna().sum()}")
print("\nFirst few rows of the dataset:")
print(df[["project_id", "project_name", "publication_date"]].head())

# Create visualizations
create_model_table()
create_timeline_visualization(df, project_fields)
create_preprocessing_wordclouds(df)
create_dataset_wordclouds(df)
create_scientific_fields_wordcloud(df)
create_training_details_wordclouds(df)
create_training_platform_wordclouds(df)
create_architecture_wordclouds(df)
create_compute_resources_wordclouds(df)
print("\nVisualizations have been saved to:")
print("- 'docs/images/model_table.svg'")
print("- 'docs/images/model_timeline.svg'")
print("- 'docs/images/preprocessing_wordclouds.svg'")
print("- 'docs/images/dataset_wordclouds.svg'")
print("- 'docs/images/scientific_fields_wordcloud.svg'")
print("- 'docs/images/training_details_wordclouds.svg'")
print("- 'docs/images/training_platform_wordclouds.svg'")
print("- 'docs/images/architecture_wordclouds.svg'")
print("- 'docs/images/compute_resources_wordclouds.svg'")
