import fire
import json
from pathlib import Path
from typing import Optional

from .collector import DataCollector
from .storage import LocalStorage

class CollectorCLI:
    def __init__(self):
        self.storage = LocalStorage()
        self.collector = DataCollector(self.storage)

    def collect(self, input_file: str, output_dir: Optional[str] = "./data"):
        """
        Collect data for LLM projects from a JSONL file.
        
        Args:
            input_file: Path to JSONL file containing project data
            output_dir: Directory to store collected data
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        with open(input_file, 'r') as f:
            for line in f:
                project_data = json.loads(line)
                self.collector.process_project(project_data)

def main():
    fire.Fire(CollectorCLI)

if __name__ == "__main__":
    main() 