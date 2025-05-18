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
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm

API_KEY = "put your openai api key here"
BIOBERT_MODEL = "dmis-lab/biobert-base-cased-v1.2"

radgraph = RadGraph()
f1radgraph = F1RadGraph(reward_level="all")
biobert_tokenizer = AutoTokenizer.from_pretrained(BIOBERT_MODEL)
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



def gpt_refined_answer(image_base64, question, prev_answer):
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": f"Question: {question}"},
            {"type": "text", "text": f"Your first answer: {prev_answer}"},
            {"type": "text", "text": (
                "Revise your initial answer."
            )},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
        ]}
    ]
    response = client.chat.completions.create(
        model="gpt-4.1-mini", messages=messages, max_tokens=500
    )
    return response.choices[0].message.content.strip()


#Medical NER model load
model_name = "blaze999/Medical-NER"
ner_tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)
ner_pipeline = pipeline("ner", model=model, tokenizer=ner_tokenizer, aggregation_strategy="simple",device=0)

def extract_entities(text):
    ner_results = ner_pipeline(text)
    entities = set([res['word'].lower() for res in ner_results])
    return entities

def calculate_f1(gt_entities, pred_entities):
    all_entities = list(gt_entities.union(pred_entities))
    if len(all_entities) == 0:
        return 0.0  
    gt_vec = [1 if e in gt_entities else 0 for e in all_entities]
    pred_vec = [1 if e in pred_entities else 0 for e in all_entities]
    _, _, f1, _ = precision_recall_fscore_support(gt_vec, pred_vec, average='binary', zero_division=0)
    return f1

scores_gt_vs_a1 = []
scores_gt_vs_a2 = []

def compute_biobert_similarity(answer, ground_truth):
    inputs1 = biobert_tokenizer(answer, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
    inputs2 = biobert_tokenizer(ground_truth, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
    with torch.no_grad():
        outputs1 = biobert(**inputs1).last_hidden_state.mean(dim=1).cpu()
        outputs2 = biobert(**inputs2).last_hidden_state.mean(dim=1).cpu()
    return 1 - cosine(outputs1.numpy().flatten(), outputs2.numpy().flatten())


results = []
output_path = "./log/final/gpt/gpt-4.1-mini_baseline_results.json"

for idx, item in enumerate(data):
    print(f"Processing {idx+1}/{len(data)}: {item['id']}")
    image = item["image"]
    question = item["question"]
    gt = item["answer"]

    try:
        answer1 = gpt_answer(image, question)
        answer2 = gpt_refined_answer(image, question, answer1)

        #a1_radgraph_f1, a1_reward_list, -, - = f1radgraph(hyps=[answer1], refs=[gt])
        a1_radgraph_f1, a1_reward_list, _, _ = f1radgraph(hyps=[answer1], refs=[gt])
        a2_radgraph_f1, a2_reward_list, _, _ = f1radgraph(hyps=[answer2], refs=[gt])

        P1_bert, _, F1_bert = bert_score([answer1], [gt], model_type="bert-base-uncased", lang="en")
        P2_bert, _, F2_bert = bert_score([answer2], [gt], model_type="bert-base-uncased", lang="en")
        
        gt_entities = extract_entities(gt)
        answer1_entities = extract_entities(answer1)
        answer2_entities = extract_entities(answer2)
        
        score_gt_vs_a1 = calculate_f1(gt_entities, answer1_entities)
        score_gt_vs_a2 = calculate_f1(gt_entities, answer2_entities)

        bio1 = compute_biobert_similarity(answer1, gt)
        bio2 = compute_biobert_similarity(answer2, gt)

        results.append({
            "id": item["id"],
            "image": image,
            "question": question,
            "ground_truth": gt,
            "answer1": answer1,
            "answer2": answer2,
            "a1_radgraph_f1": float(a1_radgraph_f1[2]),
            "a2_radgraph_f1": float(a2_radgraph_f1[2]),
            #"a2_radgraph_reward_list": a2_reward_list,
            "bert_score_answer1": float(F1_bert.mean().item()),
            "bert_score_answer2": float(F2_bert.mean().item()),
            "bert_score_improvement": float(F2_bert.mean().item() - F1_bert.mean().item()),
            "biobert_answer1": float(bio1),
            "biobert_answer2": float(bio2),
            "biobert_improvement": float(bio2 - bio1),
            "topic_similarity_score_gt_vs_answer1" : float(score_gt_vs_a1),
            "topic_similarity_score_gt_vs_answer2" : float(score_gt_vs_a2)
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
    "avg_bert_score_improvement_rate" : (avg("bert_score_answer2") - avg("bert_score_answer1") ) / avg("bert_score_answer1") * 100,
    "avg_biobert_answer1": avg("biobert_answer1"),
    "avg_biobert_answer2": avg("biobert_answer2"),
    "avg_biobert_improvement": avg("biobert_improvement"),
    "avg_biobert_improvement_rate": (avg("biobert_answer2")-avg("biobert_answer1"))/avg("biobert_answer1") * 100,
    "avg_radgraph_answer1": avg("a1_radgraph_f1"),
    "avg_radgraph_answer2": avg("a2_radgraph_f1"),
    "avg_radgraph_improvement" : (avg("a2_radgraph_f1")-avg("a1_radgraph_f1"))/avg("a1_radgraph_f1") * 100,
    "avg_topicSimilarity_answer1": avg("topic_similarity_score_gt_vs_answer1"),
    "avg_topicSimilarity_answer2": avg("topic_similarity_score_gt_vs_answer2"),
    "avg_topicSimilarity_improvement" : (avg("topic_similarity_score_gt_vs_answer2")-avg("topic_similarity_score_gt_vs_answer1"))/avg("topic_similarity_score_gt_vs_answer1") * 100,
}

with open("./log/final/gpt/gpt-4.1-mini_baseline_summary.json", "w") as f:
    json.dump(summary, f, indent=4)
print("Saved average improvement metrics to ./log/baseline/gpt_baseline_summary.json")
