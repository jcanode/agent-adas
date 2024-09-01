from typing import Dict, Any
from agent_framework import BaseAgent
import utils

class InnovativeAgent(BaseAgent):
    def __init__(self, model: str):
        super().__init__(model)
        self.reflection_steps = 5  # Number of self-reflection steps
        self.chain_length = 5  # Number of chain-of-thought steps
        self.iteration_count = 3  # Number of iterations

    def forward(self, task: Dict[str, Any]) -> str:
        best_answer = ""
        best_score = float('-inf')

        for _ in range(self.iteration_count):
            # Initial prompt
            prompt = f"Task: {task['question']}\nThink step by step:"
            thoughts = utils.query_ollama(prompt)

            # Chain-of-thought loop
            for _ in range(self.chain_length):
                chain_prompt = f"Task: {task['question']}\nThoughts: {thoughts}\nContinue the chain of thought:"
                chain_thought = utils.query_ollama(chain_prompt)
                thoughts = f"{thoughts}\nChain of thought: {chain_thought}"

            # Self-reflection loop
            for _ in range(self.reflection_steps):
                reflection_prompt = f"Task: {task['question']}\nThoughts: {thoughts}\nReflect on your thoughts and provide an improved approach:"
                reflection = utils.query_ollama(reflection_prompt)
                thoughts = f"{thoughts}\nReflection: {reflection}"

            # Final answer
            final_prompt = f"Task: {task['question']}\nThoughts: {thoughts}\nFinal answer:"
            answer = utils.query_ollama(final_prompt)

            # Score the answer
            score = utils.score_answer(task, answer)

            # Update the best answer if the current one is better
            if score > best_score:
                best_answer = answer
                best_score = score

        return best_answer