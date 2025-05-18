import os
import json
import base64
import io
from PIL import Image
from datasets import Dataset


dataset_path = "your dataset path"
arrow_file = f"{dataset_path}/med_trinity-25_m-train-00001-of-00018.arrow"

dataset = Dataset.from_file(arrow_file)

total_samples = len(dataset)
print(f"total sample #: {total_samples}")

print("\n현재 데이터 상위 3개:")
for i in range(min(3, total_samples)):
    print(dataset[i])

def process_sample(sample):
    if "image" in sample and sample["image"] is not None:
        resized_image = sample["image"].resize((256, 256), Image.Resampling.LANCZOS)
        
        buffer = io.BytesIO()
        resized_image.save(buffer, format="JPEG") 
        buffer.seek(0)

        sample["image"] = base64.b64encode(buffer.read()).decode("utf-8")
    
    return sample

json_data = [process_sample(dataset[i]) for i in range(total_samples)]


output_json_path = os.path.join(os.getcwd(), "25M_demo_2.json")
with open(output_json_path, "w") as json_file:
    json.dump(json_data, json_file, indent=4)

print(f"\noutput path: {output_json_path}")


