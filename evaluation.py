import logging
from typing import Dict, Any
from utils import query_claude, query_ollama

logger = logging.getLogger(__name__)

class Evaluator:
    def __init__(self, use_claude=True):
        self.use_claude = use_claude

    def evaluate_solution(self, task: Dict[str, Any], solution: str):
        scores = {}
        evaluation_criteria = [
            "Correctness",
            "Accuracy"
        ]

        for criterion in evaluation_criteria:
            prompt = f"""
            Evaluate the following solution for the task: "{task['question']}"

            Solution:
            {solution}

            Correct Answer:
            {task['answer']}

            Evaluation Criterion: {criterion}

            Rate the solution on a scale of 0 to 10 for this criterion.
            Provide your rating in the format "Score: X" where X is a number between 0 and 10.
            Then, provide a brief explanation for your rating.
            """
            logger.debug(f"Sending evaluation prompt for criterion: {criterion}")
            response = query_claude(prompt) if self.use_claude else query_ollama(prompt)
            score, explanation = self.parse_evaluation(response)
            scores[criterion] = {"score": score, "explanation": explanation}
            logger.info(f"Evaluation for {criterion}: Score = {score}")

        return scores

    def parse_evaluation(self, response: str):
        logger.debug(f"Raw response to parse: {response}")

        lines = response.strip().split('\n')
        score = None
        explanation_lines = []

        for line in lines:
            logger.debug(f"Processing line: {line}")
            if ':' in line and score is None:
                parts = line.split(':')
                if len(parts) >= 2:
                    try:
                        potential_score = float(parts[1].strip().split()[0])
                        if 0 <= potential_score <= 10:  # Assuming scores are between 0 and 10
                            score = potential_score
                            logger.debug(f"Found score: {score}")
                            continue
                    except ValueError:
                        logger.debug(f"Failed to parse score from: {parts[1]}")
            explanation_lines.append(line)

        if score is None:
            logger.warning(f"Failed to parse score from response. Using default score of 5.")
            score = 5  # Default middle score if parsing fails

        explanation = '\n'.join(explanation_lines).strip()
        logger.debug(f"Final parsed result - Score: {score}, Explanation: {explanation[:100]}...")

        return score, explanation

    def calculate_overall_score(self, scores: Dict[str, Dict[str, Any]]):
        return sum(score["score"] for score in scores.values()) / len(scores)

    def format_results(self, task: Dict[str, Any], solution: str, scores: Dict[str, Dict[str, Any]]):
        overall_score = self.calculate_overall_score(scores)
        result = f"Task: {task['question']}\n\n"
        result += f"Generated Solution:\n{solution}\n\n"
        result += f"Correct Answer:\n{task['answer']}\n\n"
        result += f"Overall Score: {overall_score:.2f}/10\n\n"
        result += "Detailed Scores:\n"
        for criterion, score_data in scores.items():
            result += f"{criterion}: {score_data['score']:.2f}/10\n"
            result += f"Explanation: {score_data['explanation']}\n\n"
        return result
