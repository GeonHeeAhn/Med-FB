import openai
import json
import pandas as pd
from tqdm import tqdm


openai.api_key = "put your openai api key here"

EVAL_PROMPT_TEMPLATE = """
You are a strict medical evaluator specialized in radiology.
Follow these hard constraints exactly.

Given:
- Question: {question}
- Ground Truth Answer: {ground_truth}
- Model Answer: {model_answer}

Evaluate the Model Answer according to the following criteria:

1. Relevance (20%)
2. Factuality (35%)
3. Completeness (25%)
4. Clinical Appropriateness & Conciseness (15%)
5. Fluency (5%)

Scoring Instructions:
- Each criterion must be scored from 1 to 5.
- Calculate the weighted overall score.
- All numbers must be integer scores (no half points).
- Overall score must be a floating point between 0.0 and 5.0.

Hard Output Format (strictly JSON):
{{
  "relevance": [1-5],
  "factuality": [1-5],
  "completeness": [1-5],
  "clinical_appropriateness_conciseness": [1-5],
  "fluency": [1-5],
  "overall_score": [0.0-5.0],
  "justification": "[2-3 sentence explanation]"
}}
Only output the JSON object. No additional commentary or formatting.
"""

def evaluate_answer(question, ground_truth, model_answer, model="gpt-4o"):
    prompt = EVAL_PROMPT_TEMPLATE.format(
        question=question,
        ground_truth=ground_truth,
        model_answer=model_answer
    )
    response = openai.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    content = response.choices[0].message.content.strip()

    if content.startswith("```json") or content.startswith("```"):
        content = content.split("```", 2)[1].strip()

    if content.lower().startswith("json"):
        content = content.split("\n", 1)[1].strip()

    try:
        result = json.loads(content)
        return result
    except json.JSONDecodeError:
        print("JSON Decode Error. Content received:\n", content)
        return None


def main():
    with open("./log/final/phi4/phi-4-multimodal-instruct_outputs_feedback.json", "r") as f:
        data = json.load(f)
        

    results = []

    for item in tqdm(data):
        question = item["question"]
        ground_truth = item["ground_truth"]
        initial_answer = item["answer1"]
        regenerated_answer = item["answer2"]

        initial_eval = evaluate_answer(question, ground_truth, initial_answer)

        regenerated_eval = evaluate_answer(question, ground_truth, regenerated_answer)

        if initial_eval is None or regenerated_eval is None:
            continue

        improvement = regenerated_eval["overall_score"] - initial_eval["overall_score"]

        results.append({
            "id": item["id"],
            "question": question,
            "ground_truth": ground_truth,
            "initial_answer": initial_answer,
            "initial_overall_score": initial_eval["overall_score"],
            "regenerated_answer": regenerated_answer,
            "regenerated_overall_score": regenerated_eval["overall_score"],
            "improvement": improvement,
            "initial_json": json.dumps(initial_eval),
            "regenerated_json": json.dumps(regenerated_eval),
        })

    result_df = pd.DataFrame(results)
    result_df.to_csv("./log/gpt/phi-4-multimodal-instruct_feedback_gptscore_results.csv", index=False)

    avg_initial = result_df["initial_overall_score"].mean()
    avg_regenerated = result_df["regenerated_overall_score"].mean()
    avg_improvement = result_df["improvement"].mean()
    success_rate = (result_df["improvement"] > 0).sum() / len(result_df)

    print("\n===== Evaluation Summary =====")
    print(f"Average Initial Score: {avg_initial:.4f}")
    print(f"Average Regenerated Score: {avg_regenerated:.4f}")
    print(f"Average Improvement: {avg_improvement:.4f}")
    print(f"Success Rate (Improved Cases): {success_rate * 100:.2f}%")
    
    summary = {
    "average_initial_score": avg_initial,
    "average_regenerated_score": avg_regenerated,
    "average_improvement": avg_improvement,
    "success_rate_percent": success_rate * 100
    }

    with open("./log/gpt/phi-4-multimodal-instruct_feedback_gptscore_avg.json", "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()

