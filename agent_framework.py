# agent_framework.py

from abc import ABC, abstractmethod
from typing import Dict, Any, List
import logging
# from utils import query_language_model

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    def forward(self, task: Dict[str, Any]) -> str:
        # This method should be implemented by all specific agents
        pass

    def run(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []
        for task in tasks:
            answer = self.forward(task)
            results.append({"task": task, "answer": answer})
        return results

# class SimpleAgent(BaseAgent):
#     def forward(self, task: Dict[str, Any]) -> str:
#         # Implement a simple agent logic
#         prompt = f"Task: {task['question']}\nAnswer:"
#         response = query_language_model(self.model, prompt)
#         return response

# class ChainOfThoughtAgent(BaseAgent):
#     def forward(self, task: Dict[str, Any]) -> str:
#         prompt = f"""Task: {task['question']}
# Let's approach this step-by-step:
# 1) First, let's understand the question.
# 2) Now, let's consider the relevant information.
# 3) Let's reason through the problem.
# 4) Finally, let's formulate our answer.

# Answer:"""
#         response = query_language_model(self.model, prompt)
#         return response
