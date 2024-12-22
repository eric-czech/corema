import re
import logging

logger = logging.getLogger(__name__)


def get_project_id(name: str) -> str:
    """
    Convert a project name into a standardized project ID.
    Only allows alphanumeric characters and underscores.

    Args:
        name: Project name

    Returns:
        Project ID suitable for use as a directory name and unique identifier
    """
    # Convert to lowercase and replace spaces with underscores
    name = name.lower().replace(" ", "_")

    # Remove any characters that aren't alphanumeric or underscore
    project_id = re.sub(r"[^a-z0-9_]", "", name)

    if project_id != name:
        logger.debug(f"Generated project ID '{project_id}' from name '{name}'")

    # Ensure we have a valid ID
    if not project_id:
        project_id = "unnamed_project"
        logger.warning(
            f"Project name converted to empty ID, using '{project_id}' instead"
        )

    return project_id
