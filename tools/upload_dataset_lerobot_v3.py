from lerobot.datasets import LeRobotDataset
from huggingface_hub import add_collection_item

REPO_ID = "<repo_id>"
COLLECTION = "<collection_slug>"
LOCAL_DATASET = "<file_path_to_local_dataset>"
TAGS = [
    "LeRobot",
    "RoboColosseum",
    "so101",
    "fine-tuning",
]
NOTE = "SO-101 | Pick Soccer and Place in Red Bowl | Fine-tuning demonstrations | LeRobot v3"

# Load local LeRobot v3 dataset
dataset = LeRobotDataset(
    repo_id=REPO_ID,
    root=LOCAL_DATASET,
)

# Upload
dataset.push_to_hub(
    tags=TAGS,
    upload_large_folder=True,
)

# Register in RoboColosseum collection
add_collection_item(
    collection_slug=COLLECTION,
    item_id=REPO_ID,
    item_type="dataset",
    note=NOTE,
    exists_ok=True,
)

print(f"Uploaded: https://huggingface.co/datasets/{REPO_ID}")