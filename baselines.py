from agent_framework import BaseAgent
from utils import query_ollama
from typing import List, Dict, Any
import logging
logger = logging.getLogger(__name__)

class BaselineAgent:
    def __init__(self, name: str):
        self.name = name
        self.model = "llama3"

    def forward(self, task: dict) -> str:
        raise NotImplementedError("Subclasses must implement this method")

    def run(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [{"task": task, "answer": self.forward(task)} for task in tasks]


class SimpleAgent(BaselineAgent):
    def __init__(self):
        super().__init__("Simple")

    # def forward(self, task: dict) -> str:
    #     prompt = f"Passage: {task['passage']}\nQuestion: {task['question']}\nAnswer:"
    #     return query_ollama(prompt)
    def forward(self, task: Dict[str, Any]) -> str:
        prompt = f"Task: {task['question']}\nAnswer:"
        response = query_ollama(prompt)
        if response.startswith("Error:"):
            logger.error(f"Error in SimpleAgent: {response}")
            return "Unable to generate a response due to an error."
        return response




class ChainOfThought(BaselineAgent):
    def __init__(self):
        super().__init__("Chain-of-Thought")

    def forward(self, task: dict) -> str:
        prompt = f"""Passage: {task['passage']}
Question: {task['question']}
Let's approach this step-by-step:
1) First, let's understand the question.
2) Now, let's consider the relevant information from the passage.
3) Let's reason through the problem.
4) Finally, let's formulate our answer.

Answer:"""
        response = query_ollama(prompt)
        if response.startswith("Error:"):
            logger.error(f"Error in SimpleAgent: {response}")
            return "Unable to generate a response due to an error."
        return response

class SelfRefine(BaselineAgent):
    def __init__(self):
        super().__init__("Self-Refine")

    def forward(self, task: dict) -> str:
        initial_prompt = f"Passage: {task['passage']}\nQuestion: {task['question']}\nInitial answer:"
        initial_answer = query_ollama(initial_prompt)

        refine_prompt = f"""Passage: {task['passage']}
Question: {task['question']}
Initial answer: {initial_answer}
Let's refine this answer. Consider:
1) Is the initial answer complete?
2) Are there any logical errors?
3) Can we add more relevant information?

Refined answer:"""
        response = query_ollama(refine_prompt)
        if response.startswith("Error:"):
            logger.error(f"Error in SimpleAgent: {response}")
            return "Unable to generate a response due to an error."
        return response

class LLMDebate(BaselineAgent):
    def __init__(self):
        super().__init__("LLM-Debate")

    def forward(self, task: dict) -> str:
        debate_prompt = f"""Passage: {task['passage']}
Question: {task['question']}
Let's have two AI assistants debate this question:

Assistant 1: Here's my initial thoughts on the answer...
Assistant 2: I see your point, but have you considered...
Assistant 1: That's a good point. However...
Assistant 2: Taking that into account, I think the best answer is...

Final answer based on the debate:"""
        response = query_ollama(debate_prompt)
        if response.startswith("Error:"):
            logger.error(f"Error in SimpleAgent: {response}")
            return "Unable to generate a response due to an error."
        return response

def get_baseline_agents():
    return [
        SimpleAgent(),
        ChainOfThought(),
        SelfRefine(),
        LLMDebate()
    ]
