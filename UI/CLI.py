import os
from scripts.orchestrator import Orchestrator, read_config, create_mock_db

class CLI:
    """Command-line interface for testing the VisionPipeline and AudioPipeline."""

    def __init__(self,orchestrator: Orchestrator, images_path: str) -> None:
        self.orchestrator = orchestrator
        self.images = os.listdir(images_path)
    def choose_image(self):
        print("Available images:")
        for idx, img in enumerate(self.images):
            print(f"{idx + 1}. {img}")
        choice = int(input("Select an image by number: ")) - 1
        if 0 <= choice < len(self.images):
            return os.path.join(self.images[choice])
        else:
            print("Invalid choice. Please try again.")
            return self.choose_image()
    
    def run(self):
        while True:
            image_path = self.choose_image()
            print(f"Processing {image_path}...")
            self.orchestrator.read_plate(image_path)
            cont = input("Do you want to process another image? (y/n): ")
            if cont.lower() != 'y':
                break


if __name__ == "__main__":
    config = read_config("utils/config.yaml")
    if not os.path.exists(config["database"]["db_path"]):
        create_mock_db()
    
    with Orchestrator(config, onnx=True) as orchestrator:
        cli = CLI(orchestrator, config["images"]["path"])
        cli.run()