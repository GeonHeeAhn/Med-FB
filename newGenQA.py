import openai
import base64
from PIL import Image
import io
import json
import os


API_KEY = "your openai api key"

dataset_path = "your dataset path"

print(f"Checking file at: {dataset_path}")
print(f"Absolute path: {os.path.abspath(dataset_path)}")
print(f"File size: {os.path.getsize(dataset_path)} bytes")

try:
    with open(dataset_path, "r", encoding="utf-8") as f:
        content = f.read()
        print("First 100 characters of the file:")
        print(content[:100])
        dataset = json.loads(content)
    print(f"Successfully loaded dataset with {len(dataset)} items.")
except json.JSONDecodeError as e:
    print(f"JSON decoding error: {e}")


def generate_questions(caption, image):
    prompt = f"""
Please generate ONE visual question-answer pair that strictly follow these rules strictly follow these HARD constraints:
1. You MUST use BOTH the image and the caption.
2. You MUST choose ONE and ONLY ONE of the following categories:
   - (1) Lesion Location: Where is the lesion located, and how does it relate to nearby structures?
   - (2) Shape & Size: What is the size, shape, or boundary of the lesion?
   - (3) Density/Attenuation: Describe the brightness/density/contrast of the lesion compared to nearby brain tissue.
   - (4) Mass Effect: What effect does the lesion have on midline structures, ventricles, or sulci?
   - (5) Number and Distribution: How many lesions are visible and how are they distributed?
   - (6) Modality-based Appearance: How is the lesion visualized in this CT, and how might contrast media affect its appearance?

3. DO NOT generate general radiology or medical knowledge questions.
4. DO NOT generate yes/no questions.
5. The question MUST require visual interpretation of the image. Do not rely on caption only.
6. The answer MUST be 1-3 full sentences using medical terminology.


Caption:
"{caption}"

---

Now respond ONLY in this strict JSON format:
{{
  "category": <category_number>,
  "question": "<a single image-dependent question>",
  "answer": "<an answer that depends on the visual features>"
}}

# ❌ Do NOT include anything outside this format.
# ❌ Do NOT use general radiology knowledge.
# ✅ Make sure the answer requires visual inspection of the image.

"""

    try:
        image_url = image
        if not image.startswith("data:image"):
            image_url = "data:image/jpeg;base64," + image
        client = openai.Client(api_key=API_KEY)
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You are a highly specialized assistant trained in interpreting CT/radiology images and generating image-dependent VQA pairs using strict medical reasoning. Use the base64 format image provided below."},
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }
            ],
            temperature=0.5,
            max_tokens=1500
        )
        response_content = response.choices[0].message.content
        
        #print("📦 Full model response:")
        #print(response)

        return [json.loads(response_content)]

    
    except json.JSONDecodeError as e:
        print(f"❌ JSON decoding error for ID {item['id']}: {e}")
        print(f"Raw content was:\n{response_content}")
        return []
    except Exception as e:
        print(f"Error generating questions for ID {item['id']}: {e}")
        return []

final_dataset = []
for item in dataset[1000:3000]:
    print(f"Processing ID: {item['id']}")

    try:
        img = Image.open(io.BytesIO(base64.b64decode(item["image"])))
        print(f"✅ Image decoded successfully for ID {item['id']}")
    except Exception as e:
        print(f"❌ Image decoding failed for ID {item['id']}: {e}")

    try:
        qa_pairs = generate_questions(item["caption"], item["image"])
        for qa in qa_pairs:
            result = {
                "id": item["id"],
                "image": item["image"],
                "question": qa["question"],
                "answer": qa["answer"],
                "category": qa["category"],
                "caption": item["caption"]
            }
            final_dataset.append(result)

            print(f"Generated Q&A for ID {item['id']}:")
            print(f"  Description: {item['caption']}")
            print(f"  Question: {qa['question']}")
            print(f"  Answer: {qa['answer']}")
            print("-" * 50)
    except Exception as e:
        print(f"Error generating QA for ID {item['id']}: {e}")


output_path = "./dataset/MedReflect-VQA.json"
try:
    with open(output_path, "w") as f:
        json.dump(final_dataset, f, indent=4)
    print(f"Dataset saved to {output_path}")
except Exception as e:
    print(f"Error saving dataset: {e}")

