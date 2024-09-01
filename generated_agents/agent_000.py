from typing import Dict, Any
from agent_framework import BaseAgent
import utils

class InnovativeAgent(BaseAgent):
    def __init__(self, model: str):
        super().__init__(model)
        self.reflection_steps = 3  # Number of self-reflection steps

    def forward(self, task: Dict[str, Any]) -> str:
        # Initial prompt
        prompt = f"Task: {task['question']}\nThink step by step:"
        thoughts = utils.query_ollama(prompt)

        # Self-reflection loop
        for _ in range(self.reflection_steps):
            reflection_prompt = f"Task: {task['question']}\nThoughts: {thoughts}\nReflect on your thoughts and provide an improved approach:"
            reflection = utils.query_ollama(reflection_prompt)
            thoughts = f"{thoughts}\nReflection: {reflection}"

        # Final answer
        final_prompt = f"Task: {task['question']}\nThoughts: {thoughts}\nFinal answer:"
        answer = utils.query_ollama(final_prompt)

        return answer