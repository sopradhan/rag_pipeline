from transformers import pipeline
from datetime import datetime
from incident_management.repositories import update_row_corrective_action

class CorrectiveActionGenerator:
    def __init__(self, model_name="google/flan-t5-large"):
        self.pipe = pipeline("text2text-generation", model=model_name)

    def build_prompt(self, row):
        exclude = [
            "bert_score", "rule_score", "combined_score", "id", 
            "corrective_action", "processed_at"
        ]
        prompt_data = {k: v for k, v in row.items() if k not in exclude}
        prompt = (
            "Given the following incident information, think through and suggest an actionable corrective action:\n"
        )
        for key, value in prompt_data.items():
            prompt += f"{key}: {value}\n"
        prompt += "\nCorrective Action:"
        return prompt

    def generate_and_update(self, row):
        prompt = self.build_prompt(row)
        response = self.pipe(prompt, max_new_tokens=80)[0]['generated_text'].strip()
        now_str = datetime.utcnow().isoformat(timespec='seconds')
        update_row_corrective_action(row['id'], response, now_str)
