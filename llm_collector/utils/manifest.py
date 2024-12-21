import json
from pathlib import Path
from typing import Dict, Any

class ManifestManager:
    def __init__(self, manifest_path: str = "manifest.json"):
        self.manifest_path = Path(manifest_path)
        self.manifest: Dict[str, Any] = self._load_manifest()

    def _load_manifest(self) -> Dict[str, Any]:
        if self.manifest_path.exists():
            return json.loads(self.manifest_path.read_text())
        return {}

    def add_project(self, project_name: str, metadata: Dict[str, Any]):
        """Add or update project metadata in manifest."""
        self.manifest[project_name] = metadata
        self._save_manifest()

    def _save_manifest(self):
        self.manifest_path.write_text(json.dumps(self.manifest, indent=2))

    def get_project(self, project_name: str) -> Dict[str, Any]:
        return self.manifest.get(project_name, {}) 