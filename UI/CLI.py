import os
from scripts.orchestrator import Orchestrator, read_config, create_mock_db

class CLI:
    """Command-line interface for testing the VisionPipeline and AudioPipeline."""

    def __init__(self,orchestrator: Orchestrator, image_path: str) -> None:
        self.orchestrator = orchestrator
        self.images = os
    def choose_image(self):
        pass
    
    def run(self):


if __name__ == "__main__":
    config = read_config("utils/config.yaml")
    if not os.path.exists(config["database"]["db_path"]):
        create_mock_db()
    
    with Orchestrator(config, onnx=True) as orchestrator:
        pass  # The GUI will be launched from UI/GUI.py, which imports Orchestrator.