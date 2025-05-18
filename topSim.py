import json
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm


model_name = "blaze999/Medical-NER"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple",device=0)

def extract_entities(text):
    ner_results = ner_pipeline(text)
    entities = set([res['word'].lower() for res in ner_results])
    return entities

def calculate_f1(gt_entities, pred_entities):
    all_entities = list(gt_entities.union(pred_entities))
    if len(all_entities) == 0:
        return 0.0  # avoid division by zero
    gt_vec = [1 if e in gt_entities else 0 for e in all_entities]
    pred_vec = [1 if e in pred_entities else 0 for e in all_entities]
    _, _, f1, _ = precision_recall_fscore_support(gt_vec, pred_vec, average='binary', zero_division=0)
    return f1

input_file = "./log/final/qwen2.5vl/qwen2.5-vl-32b-instruct_outputs_baseline.json"  
output_file = "./log/topicSim/qwen2.5-vl-32b-instruct_baseline_topsim.jsonl"
summary_file = "./log/topicSim/sum/qwen2.5-vl-32b-instruct_baseline_topsim_sum.json"

scores_gt_vs_a1 = []
scores_gt_vs_a2 = []

with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    data_list = json.load(infile)  # !!! json.load() 전체 로드

    for data in tqdm(data_list, desc="Processing samples"):
        
        gt_entities = extract_entities(data["ground_truth"])
        answer1_entities = extract_entities(data["answer1"])
        answer2_entities = extract_entities(data["answer2"])
        
        score_gt_vs_a1 = calculate_f1(gt_entities, answer1_entities)
        score_gt_vs_a2 = calculate_f1(gt_entities, answer2_entities)
        
        data["topic_similarity_score_gt_vs_answer1"] = score_gt_vs_a1
        data["topic_similarity_score_gt_vs_answer2"] = score_gt_vs_a2
        
        if "image" in data:
            del data["image"]

        outfile.write(json.dumps(data) + '\n')
        
        scores_gt_vs_a1.append(score_gt_vs_a1)
        scores_gt_vs_a2.append(score_gt_vs_a2)

average_score_a1 = sum(scores_gt_vs_a1) / len(scores_gt_vs_a1) if scores_gt_vs_a1 else 0
average_score_a2 = sum(scores_gt_vs_a2) / len(scores_gt_vs_a2) if scores_gt_vs_a2 else 0
average_improvement = average_score_a2 - average_score_a1

summary = {
    "average_topic_similarity_score_gt_vs_answer1": average_score_a1,
    "average_topic_similarity_score_gt_vs_answer2": average_score_a2,
    "average_topic_similarity_score_improvement": average_improvement
}

with open(summary_file, 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2)

print("Saved")
