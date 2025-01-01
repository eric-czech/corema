import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Optional

from wordcloud import WordCloud
from plotnine import (
    ggplot,
    aes,
    geom_text,
    geom_tile,
    theme,
    element_text,
    element_blank,
    labs,
    theme_minimal,
    scale_fill_distiller,
    scale_color_manual,
)
import logging

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

logger = logging.getLogger(__name__)


def create_model_table(df: pd.DataFrame, images_dir: Path) -> None:
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
    plt.savefig(images_dir / "model_table.svg", bbox_inches="tight", format="svg")
    plt.close()


def create_timeline_visualization(
    df: pd.DataFrame, project_fields: Dict[str, List[str]], images_dir: Path
) -> None:
    """Create a timeline visualization showing models and their scientific fields."""
    # Filter out models without dates
    df_with_dates = df[df["publication_date"].notna()].copy()

    # Extract parameter counts from architecture dictionary
    def get_param_count(arch: Dict[str, Any]) -> Optional[float]:
        """Extract parameter count from architecture dictionary."""
        if isinstance(arch, dict) and "parameter_count" in arch:
            counts = arch["parameter_count"]
            if isinstance(counts, (list, np.ndarray)) and len(counts) > 0:
                # Convert to numeric and get max, handling non-numeric values
                numeric_counts = pd.to_numeric(pd.Series(counts), errors="coerce")
                max_count = float(numeric_counts.max())  # Explicitly convert to float
                # Ignore counts less than 1M
                if pd.notna(max_count) and max_count >= 1e6:
                    return max_count
        return None

    parameter_counts = df_with_dates["architecture"].apply(get_param_count)

    # Calculate dot sizes based on parameter counts
    # Use log scale for better visualization, add 1 to handle zeros
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
    plt.figure(figsize=(14, len(df_with_dates) * 0.5))
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
    plt.savefig(images_dir / "model_timeline.svg", bbox_inches="tight", format="svg")
    plt.close()


def create_affiliation_visualization(df: pd.DataFrame, images_dir: Path) -> None:
    """Create visualization showing the frequency of affiliations across papers."""
    # Load affiliation mapping from CSV in resources directory
    mapping_path = Path(__file__).parent / "resources" / "affiliations.csv"
    if not mapping_path.exists():
        raise FileNotFoundError(
            f"Affiliations CSV not found at {mapping_path!r}. Please run the create_affiliations_csv.py script first."
        )

    affiliations_df = pd.read_csv(mapping_path)
    affiliation_mapping = dict(
        zip(affiliations_df["raw_affiliation"], affiliations_df["normalized_name"])
    )

    # Create dictionary to track projects per affiliation
    affiliation_projects: Dict[str, set[str]] = {}

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
    plot_df = pd.DataFrame(
        [
            {
                "affiliation": affiliation,
                "count": len(projects),
                "category": affiliations_df[
                    affiliations_df["normalized_name"] == affiliation
                ]["category"].iloc[0],
                "subcategory": affiliations_df[
                    affiliations_df["normalized_name"] == affiliation
                ]["subcategory"].iloc[0],
            }
            for affiliation, projects in affiliation_projects.items()
        ]
    )

    # Create and clean up group labels
    plot_df["category"] = plot_df["category"].str.replace("_", " ").str.title()
    plot_df["subcategory"] = plot_df["subcategory"].str.replace("_", " ").str.title()
    plot_df["group"] = plot_df["category"] + " / " + plot_df["subcategory"]

    # Order groups by number of affiliations
    group_sizes = plot_df.groupby("group").size().sort_values(ascending=False)
    plot_df["group"] = pd.Categorical(plot_df["group"], categories=group_sizes.index)

    # Sort within each group by count and add order
    plot_df = plot_df.sort_values(["group", "count"], ascending=[True, True])
    plot_df["order"] = plot_df.groupby("group").cumcount() + 1

    # Clip affiliation names to 40 chars
    plot_df["affiliation_label"] = plot_df["affiliation"].apply(
        lambda x: x if len(x) <= 40 else x[:37] + "..."
    )

    # Create heatmap using plotnine
    plot = (
        ggplot(plot_df, aes(y="order", x="group"))
        + geom_tile(aes(fill="count"), color="black", size=0.25)
        + scale_fill_distiller(type="seq", palette="Blues", direction=1)
        + geom_text(
            aes(label="affiliation_label", color="count > 3"),  # Use boolean condition
            size=6,
            ha="center",
            va="center",
        )
        + scale_color_manual(
            values={True: "white", False: "black"},
            guide=None,  # Hide the color legend for text
        )
        + labs(
            title="Organizations in Foundation Model Papers",
            y="Order",
            x="Category/Type",
            fill="Number of\nProjects",
        )
        + theme_minimal()
        + theme(
            figure_size=(18, 10),
            plot_title=element_text(size=12, face="bold"),
            axis_text_x=element_text(
                size=8,
                angle=30,
                ha="right",
                va="top",
                face="bold",
                vjust=1,  # Move labels down more
            ),
            axis_text_y=element_text(size=8),
            axis_title=element_text(size=10),
            legend_position="right",
            panel_grid_major=element_blank(),  # Remove grid lines
            panel_grid_minor=element_blank(),
        )
    )

    # Save plot
    plot.save(images_dir / "affiliations.svg", dpi=300, limitsize=False)


def create_category_wordclouds(
    df: pd.DataFrame,
    column: str,
    categories: List[str],
    title_prefix: str,
    output_file: str,
    images_dir: Path,
    n_cols: int = 2,
) -> None:
    """Create word cloud visualizations for categories in a column."""
    # Create a figure with subplots for each category
    n_rows = (len(categories) + n_cols - 1) // n_cols
    plt.figure(figsize=(10 * n_cols, 5 * n_rows))

    for idx, category in enumerate(categories, 1):
        # Create a dictionary to track which projects mention each item
        item_projects: Dict[str, set[str]] = {}

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
    output_path = images_dir / output_file
    plt.savefig(output_path, bbox_inches="tight", format="svg")
    plt.close()


def create_preprocessing_wordclouds(df: pd.DataFrame, images_dir: Path) -> None:
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
        images_dir=images_dir,
    )


def create_dataset_wordclouds(df: pd.DataFrame, images_dir: Path) -> None:
    """Create word cloud visualizations for dataset categories."""
    categories = ["modalities"]
    create_category_wordclouds(
        df=df,
        column="dataset",
        categories=categories,
        title_prefix="Dataset",
        output_file="dataset_wordclouds.svg",
        images_dir=images_dir,
        n_cols=1,
    )


def create_scientific_fields_wordcloud(df: pd.DataFrame, images_dir: Path) -> None:
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
        images_dir=images_dir,
        n_cols=1,
    )


def create_training_details_wordclouds(df: pd.DataFrame, images_dir: Path) -> None:
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
        images_dir=images_dir,
    )


def create_training_platform_wordclouds(df: pd.DataFrame, images_dir: Path) -> None:
    """Create word cloud visualizations for training platform categories."""
    categories = ["cloud_provider", "hpc_system", "training_service", "other_platforms"]
    create_category_wordclouds(
        df=df,
        column="training_platform",
        categories=categories,
        title_prefix="Training Platform",
        output_file="training_platform_wordclouds.svg",
        images_dir=images_dir,
    )


def create_architecture_wordclouds(df: pd.DataFrame, images_dir: Path) -> None:
    """Create word cloud visualizations for architecture categories."""
    categories = ["model_type", "architecture_type", "key_components"]
    create_category_wordclouds(
        df=df,
        column="architecture",
        categories=categories,
        title_prefix="Architecture",
        output_file="architecture_wordclouds.svg",
        images_dir=images_dir,
    )


def create_compute_resources_wordclouds(df: pd.DataFrame, images_dir: Path) -> None:
    """Create word cloud visualizations for compute resources categories."""
    categories = ["training_time", "cost_estimate", "gpu_type", "other_hardware"]
    create_category_wordclouds(
        df=df,
        column="compute_resources",
        categories=categories,
        title_prefix="Compute Resources",
        output_file="compute_resources_wordclouds.svg",
        images_dir=images_dir,
    )
