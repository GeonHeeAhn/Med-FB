import openai
import json
import os
import numpy as np
from radgraph import RadGraph, F1RadGraph
from transformers import AutoTokenizer, AutoModel
from bert_score import score as bert_score
from nltk.tokenize import word_tokenize
import torch
from scipy.spatial.distance import cosine
from statistics import mean

API_KEY = "your openai api key here"
BIOBERT_MODEL = "dmis-lab/biobert-base-cased-v1.2"

radgraph = RadGraph()
f1radgraph = F1RadGraph(reward_level="all")
tokenizer = AutoTokenizer.from_pretrained(BIOBERT_MODEL)
biobert = AutoModel.from_pretrained(BIOBERT_MODEL).to("cuda")


input_file = "./dataset/MedReflect-VQA.json"
with open(input_file, "r") as f:
    data = json.load(f)


client = openai.Client(api_key=API_KEY)


def gpt_answer(image_base64, question):
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": f"Question: {question}"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
        ]}
    ]
    response = client.chat.completions.create(
        model="gpt-4.1-mini", messages=messages, max_tokens=500
    )
    return response.choices[0].message.content.strip()

def gpt_feedback(image_base64, question, answer, ground_truth):
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": f"Question: {question}"},
            {"type": "text", "text": f"Student Answer: {answer}"},
            {"type": "text", "text" : f"Ground Truth: {ground_truth}"},
            {"type": "text", "text": 
                """You are assisting a medical student in refining their answer based on a radiology and CT image.

Your main objective is to help the student **semantically align** their answer more closely with the ideal expert-level answer, based on both the question and the image.

Guidelines:
- **Preserve** any parts of the student's initial answer that are already semantically correct and visually aligned.
- Identify and point out **missing or vague critical medical details** (e.g., anatomical structures, imaging characteristics, clinical implications) that an expert would include.
- Suggest **specific improvements** that enhance semantic completeness and expert precision.
- Avoid suggesting unnecessary stylistic rephrasing or speculative content not grounded in the image.

Instructions for Feedback:
- Write exactly 2 informative sentences.
    - The first should **highlight what important information is missing or vague**.
    - The second should **propose a specific addition or clarification** that would semantically strengthen the answer.
- Focus on **completeness, accuracy, and expert-level terminology**, while keeping the response concise and grounded in the visual information.

                """
            },
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
        ]}
    ]
    response = client.chat.completions.create(
        model="gpt-4.1-mini", messages=messages, max_tokens=300
    )
    return response.choices[0].message.content.strip()

def gpt_refined_answer(image_base64, question, feedback, prev_answer):
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": f"Feedback: {feedback}"},
            {"type": "text", "text": f"Question: {question}"},
            {"type": "text", "text": f"Your first answer: {prev_answer}"},
            {"type": "text", "text": (
                """Revise your initial answer by fully incorporating the feedback.

- Add any important anatomical, imaging, or clinical details mentioned.
- Be specific and detailed, ensuring your revised answer aligns closely with the correct interpretation.
- Write 2-3 concise, medically accurate sentences based on the provided feedback and the image.

Your revised answer should show significant improvement toward the ideal correct answer.

                """
            )},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
        ]}
    ]
    response = client.chat.completions.create(
        model="gpt-4.1-mini", messages=messages, max_tokens=500
    )
    return response.choices[0].message.content.strip()


def compute_biobert_similarity(answer, ground_truth):
    inputs1 = tokenizer(answer, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
    inputs2 = tokenizer(ground_truth, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
    with torch.no_grad():
        outputs1 = biobert(**inputs1).last_hidden_state.mean(dim=1).cpu()
        outputs2 = biobert(**inputs2).last_hidden_state.mean(dim=1).cpu()
    return 1 - cosine(outputs1.numpy().flatten(), outputs2.numpy().flatten())


results = []
output_path = "./log/final/gpt/results.json"

for idx, item in enumerate(data):
    print(f"Processing {idx+1}/{len(data)}: {item['id']}")
    image = item["image"]
    question = item["question"]
    gt = item["answer"]

    try:
        answer1 = gpt_answer(image, question)
        feedback = gpt_feedback(image, question, answer1, gt)
        answer2 = gpt_refined_answer(image, question, feedback, answer1)

        a1_radgraph_f1, f1_reward_list, _, _ = f1radgraph(hyps=[answer1], refs=[gt])
        fb_radgraph_f1, fb_reward_list, _, _ = f1radgraph(hyps=[feedback], refs=[gt])
        a2_radgraph_f1, a2_reward_list, _, _ = f1radgraph(hyps=[answer2], refs=[gt])

        P1_bert, _, F1_bert = bert_score([answer1], [gt], model_type="bert-base-uncased", lang="en")
        P2_bert, _, F2_bert = bert_score([answer2], [gt], model_type="bert-base-uncased", lang="en")


        bio1 = compute_biobert_similarity(answer1, gt)
        bio2 = compute_biobert_similarity(answer2, gt)

        results.append({
            "id": item["id"],
            "image": image,
            "question": question,
            "ground_truth": gt,
            "answer1": answer1,
            "feedback": feedback,
            "answer2": answer2,
            "a1_radgraph_f1": a1_radgraph_f1[2],
            "fb_radgraph_f1": fb_radgraph_f1[2],
            "a2_radgraph_f1": a2_radgraph_f1[2],
            "bert_score_answer1": F1_bert.mean().item(),
            "bert_score_answer2": F2_bert.mean().item(),
            "bert_score_improvement": F2_bert.mean().item() - F1_bert.mean().item(),
            "biobert_answer1": bio1,
            "biobert_answer2": bio2,
            "biobert_improvement": bio2 - bio1,
        })

    except Exception as e:
        print(f"Error processing {item['id']}: {e}")

with open(output_path, "w") as f:
    json.dump(results, f, indent=4)
print(f"Results saved to {output_path}")

def avg(key):
    return mean([r[key] for r in results if key in r])

summary = {
    "avg_bert_score_answer1": avg("bert_score_answer1"),
    "avg_bert_score_answer2": avg("bert_score_answer2"),
    "avg_bert_score_improvement": avg("bert_score_improvement"),
    "avg_biobert_answer1": avg("biobert_answer1"),
    "avg_biobert_answer2": avg("biobert_answer2"),
    "avg_biobert_improvement": avg("biobert_improvement"),
    "avg_radgraph_a1": avg("a1_radgraph_f1"),
    "avg_radgraph_fb": avg("fb_radgraph_f1"),
    "avg_radgrpach_answer2": avg("a2_radgraph_f1"),
}

with open("./log/final/gpt/summary.json", "w") as f:
    json.dump(summary, f, indent=4)
print("Saved average improvement metrics to ./log/final/gpt/summary.json")
