import json
from tqdm import tqdm
import openai
import re

API_KEY = "your openai api key here"

input_file = "./dataset/MedReflect-VQA.json"
output_file = "./dataset/scorewise_difficult_questions.json"

def get_difficulty_score(question, caption, image):
    prompt = f"""
You are a radiologist and an expert in medical education, specializing in evaluating the difficulty of medical visual question answering (VQA) tasks.

You will be given the following information about a VQA task:
1. A natural language **question** referring to a medical image.
2. A **caption** describing the medical image.
3. A **base64-encoded image** representing the medical image.

Please assess the overall difficulty of answering this question based on the image and caption, by considering the following factors:
- The level of **radiological knowledge** required to answer the question.
- The level of **medical reasoning or inference** required beyond direct observation.
- The level of **image interpretation** skill required to identify and integrate findings from the image.
- The **complexity of the anatomical region** and radiological findings involved.
- The **specificity and scope** of the question (e.g., general description vs. very specific diagnostic feature).

IMPORTANT INSTRUCTIONS:

- A score of **4 to  5 should be very rare** and should only be assigned to questions that would challenge **highly experienced radiologists or require subtle, complex interpretation and reasoning**.
- A score of **1 to  2 should be used when a general radiology student or junior resident could answer the question confidently**.
- Please avoid rating every question as "moderate" or "hard"; instead, differentiate carefully based on these definitions.
- Be as precise as possible in your scoring.
- Return **only a single numeric score as a floating point number (e.g., 4.5, 4.7, 5.0).**
- Do not include any explanations, units, symbols, or additional text.
- Analyze the question, caption, and image description holistically.
- Focus on the medical expertise, reasoning, and image analysis required.


Here is the difficulty rating scale:

1.0 = Very easy (can be answered by basic visual recognition; minimal medical knowledge required)
2.0 = Easy (requires basic radiological knowledge and simple interpretation)
3.0 = Moderate (requires intermediate radiological knowledge and some integration of findings)
4.0 = Hard (requires advanced radiological knowledge, detailed interpretation, and clinical reasoning)
5.0 = Very hard (requires expert-level radiological knowledge, subtle interpretation, and complex clinical reasoning)

You are allowed to rate the difficulty using decimal values between 1.0 and 5.0 (e.g., 3.5, 4.2, 4.7), depending on nuance.

Here is the VQA task:

Question: {question}

Caption: {caption}

Base64-encoded image: {image}

Please provide only the floating point score in your response.
"""
    try:
        client = openai.Client(api_key=API_KEY)
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}}
                    ]
                }
            ],
            temperature=0.5,
            max_tokens=15
        )
        response_content = response.choices[0].message.content

        return [json.loads(response_content)]


    except json.JSONDecodeError as e:
        print(f"❌ JSON decoding error for ID {item['id']}: {e}")
        print(f"Raw content was:\n{response_content}")
        return []
    except Exception as e:
        print(f"Error generating questions for ID {item['id']}: {e}")
        return []
    
with open(input_file, 'r', encoding='utf-8') as infile:
    data = json.load(infile)

hard_questions = []

for item in tqdm(data, desc="Evaluating difficulty"):
    try:
        score_raw = get_difficulty_score(item['question'], item['caption'], item['image'])
        print(f"ID {item['id']} - Raw output: {score_raw}")
        
        match = re.search(r'\d+(\.\d+)?', str(score_raw))
        if match:
            score = float(match.group(0))
            print(f"ID {item['id']} - Parsed score: {score}")

            if score >= 4.5:
                hard_questions.append(item)
        else:
            print(f"ID {item['id']} - No valid score found in output: {score_raw}")
    
    except Exception as e:
        print(f"Error processing ID {item['id']}: {e}")


with open(output_file, 'w', encoding='utf-8') as outfile:
    json.dump(hard_questions, outfile, indent=2, ensure_ascii=False)

print(f"총 {len(hard_questions)}개의 hard questions가 저장되었습니다.")

results = []

for item in tqdm(data, desc="Evaluating difficulty"):
    try:
        score_raw = get_difficulty_score(item['question'], item['caption'], item['image'])
        print(f"ID {item['id']} - Raw output: {score_raw}")

        match = re.search(r'\d+(\.\d+)?', str(score_raw))
        if match:
            score_parsed = float(match.group(0))
            print(f"ID {item['id']} - Parsed score: {score_parsed}")
        else:
            score_parsed = None
            print(f"ID {item['id']} - No valid score found in output: {score_raw}")

        item_result = item.copy()  
        item_result['score_raw'] = score_raw
        item_result['score_parsed'] = score_parsed
        results.append(item_result)

    except Exception as e:
        print(f"Error processing ID {item['id']}: {e}")
        item_result = item.copy()
        item_result['score_raw'] = None
        item_result['score_parsed'] = None
        item_result['error'] = str(e)
        results.append(item_result)


output_filename = "./dataset/scorewise_3000.json"
with open(output_filename, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print(f"Saved results to {output_filename}")
