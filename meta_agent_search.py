import random
from typing import List, Dict, Any
from agent_framework import BaseAgent
from baselines import BaselineAgent, SimpleAgent
from utils import evaluate_performance, save_results, query_ollama, query_claude
import os
import json
from datetime import datetime
import re
import ast
import logging
import importlib

logger = logging.getLogger(__name__)

class MetaAgentSearch:
    def __init__(self, meta_agent, max_iterations: int = 3):
        self.meta_agent = meta_agent
        self.max_iterations = max_iterations
        self.archive = []
        self.agent_dir = "generated_agents"
        os.makedirs(self.agent_dir, exist_ok=True)

    def generate_new_agent(self) -> str:
        prompt = f"""Based on the following archive of agents:
    {self.format_archive()}

    Generate a new, innovative agent that could potentially outperform the existing ones.
    Provide a complete Python class definition for the new agent.

    Guidelines:
    1. The class should inherit from BaseAgent.
    2. Include necessary imports at the top (from typing import Dict, Any; from agent_framework import BaseAgent).
    3. Implement __init__ and forward methods.
    4. Use utils.query_ollama(prompt) for language model interactions instead of self.model.generate.
    5. Focus on high-level logic and decision-making.
    6. Be innovative - consider techniques like chain-of-thought, self-reflection, or multi-step reasoning.

    Example structure:

    from typing import Dict, Any
    from agent_framework import BaseAgent
    import utils

    class InnovativeAgent(BaseAgent):
        def __init__(self, model: str):
            super().__init__(model)
            # Additional initialization if needed

        def forward(self, task: Dict[str, Any]) -> str:
            # Implement your innovative approach here
            # Example:
            prompt = f"Task: {{task['question']}}\nThink step by step:"
            thoughts = utils.query_ollama(prompt)

            final_prompt = f"Task: {{task['question']}}\nThoughts: {{thoughts}}\nFinal answer:"
            answer = utils.query_ollama(final_prompt)

            return answer

    Start your code with ```python and end it with ```.
    Provide only the Python code, without any additional explanations.
    """
        response = self.meta_agent.generate_agent(prompt)

        # Ensure the response starts with imports
        # if response.find("from typing") >= -1:
        # # if not response.startswith("from typing"):
        #     response = "from typing import Dict, Any\nfrom agent_framework import BaseAgent\nimport utils\n\n" + response
        code_start = response.find("```python")
        code_end = response.rfind("```")
        if code_start != -1 and code_end != -1:
            code = response[code_start+9:code_end].strip()
            return code
        else:
            raise ValueError("No valid Python code block found in the generated response")
            return response

    def save_agent(self, agent_code: str, performance: float, iteration: int):
        # Save agent code
        agent_filename = f"agent_{iteration:03d}_perf_{performance:.4f}.py"
        with open(os.path.join(self.agent_dir, agent_filename), 'w') as f:
            f.write(agent_code)

        # Save metadata
        metadata = {
            "iteration": iteration,
            "performance": performance,
            "timestamp": datetime.now().isoformat()
        }
        metadata_filename = f"agent_{iteration:03d}_metadata.json"
        with open(os.path.join(self.agent_dir, metadata_filename), 'w') as f:
            json.dump(metadata, f, indent=2)

    def format_archive(self) -> str:
        return "\n\n".join([f"Agent {i+1}:\n{agent}" for i, agent in enumerate(self.archive)])

    def evaluate_agent(self, agent_code: str, tasks: List[Dict[str, Any]]) -> float:
        results = []
        for task in tasks:
            prompt = f"""
Use the following agent code to answer the question:

{agent_code}

Passage: {task['passage']}
Question: {task['question']}
Answer:"""
            # answer = query_ollama(prompt)
            # results.append({"task": task, "answer": answer})
            print("Prompt sent to Ollama:")
            print(prompt)
            try:
                answer = query_ollama(prompt)
                results.append({"task": task, "answer": answer})
            except Exception as e:
                print(f"Error querying Ollama: {e}")
                results.append({"task": task, "answer": "Error"})

        correct = sum(1 for r in results if r['task']['answer'].strip().lower() in r['answer'].strip().lower())
        return correct / len(results)

    def search(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        best_agent = None
        best_performance = 0
        best_agent_results = []

        for iteration in range(self.max_iterations):
            try:
                new_agent_code = self.generate_new_agent()
                # validated_code = self.validate_and_clean_code(new_agent_code)

                # Save the agent code
                agent_filename = f"agent_{iteration:03d}.py"
                with open(os.path.join(self.agent_dir, agent_filename), 'w') as f:
                    f.write(new_agent_code)

                # Dynamically load and instantiate the agent
                spec = importlib.util.spec_from_file_location(f"agent_{iteration}", os.path.join(self.agent_dir, agent_filename))
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Find the agent class (should be the only class inheriting from BaseAgent)
                agent_class = next(obj for name, obj in module.__dict__.items()
                                   if isinstance(obj, type) and issubclass(obj, BaseAgent) and obj != BaseAgent)
                new_agent = agent_class(self.meta_agent.model)  # Changed from self.llm_interface.model

                performance = self.evaluate_agent(new_agent, tasks)

                # Update metadata
                metadata = {
                    "iteration": iteration,
                    "performance": performance,
                    "timestamp": datetime.now().isoformat(),
                    "agent_name": agent_class.__name__
                }
                with open(os.path.join(self.agent_dir, f"agent_{iteration:03d}_metadata.json"), 'w') as f:
                    json.dump(metadata, f, indent=2)

                if performance > best_performance:
                    best_agent = new_agent
                    best_performance = performance
                    best_agent_results = [{"task": task, "answer": new_agent.forward(task)} for task in tasks]

                # Add to archive
                self.archive.append(new_agent_code)

            except Exception as e:
                logger.error(f"Error creating or evaluating new agent: {e}")

        return {
            "best_agent": best_agent,
            "best_performance": best_performance,
            "best_agent_results": best_agent_results,
            "archive": self.archive
        }



    # # def validate_and_clean_code(self, code: str) -> str:
    #     try:
    #         # Ensure necessary imports are present
    #         if "from typing import Dict, Any" not in code:
    #             code = "from typing import Dict, Any\n" + code
    #         if "from agent_framework import BaseAgent" not in code:
    #             code = "from agent_framework import BaseAgent\n" + code
    #         if "import utils" not in code:
    #             code = "import utils\n" + code

    #         # Parse the code into an AST
    #         tree = ast.parse(code)

    #         # Check if there's at least one class definition
    #         if not any(isinstance(node, ast.ClassDef) for node in ast.walk(tree)):
    #             raise ValueError("No class definition found in the generated code")

    #         # Check if the class inherits from BaseAgent
    #         class_def = next(node for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
    #         if not any(base.id == 'BaseAgent' for base in class_def.bases if isinstance(base, ast.Name)):
    #             raise ValueError("Generated class does not inherit from BaseAgent")

    #         # Check for __init__ and forward methods
    #         methods = {node.name for node in ast.walk(class_def) if isinstance(node, ast.FunctionDef)}
    #         if '__init__' not in methods or 'forward' not in methods:
    #             raise ValueError("Generated class is missing __init__ or forward method")

    #         # Check for correct use of utils.query_ollama
    #         if "utils.query_ollama" not in code:
    #             raise ValueError("Generated code does not use utils.query_ollama for language model interactions")

    #         # The code is valid, return it as-is
    #         return code
    #     except SyntaxError as e:
    #         # If there's a syntax error, try to fix common issues
    #         lines = code.split('\n')
    #         # Remove any lines that don't look like valid Python code
    #         cleaned_lines = [line for line in lines if line.strip() and not line.strip().startswith(('#', '//'))]
    #         cleaned_code = '\n'.join(cleaned_lines)

    #         # Try parsing again
    #         try:
    #             ast.parse(cleaned_code)
    #             return cleaned_code
    #         except SyntaxError:
    #             raise ValueError("Unable to generate valid Python code")

# def main():
#     # Sample tasks (replace with actual tasks from your dataset)
#     tasks = [
#         {"question": "What is 2 + 2?", "correct_answer": "4"},
#         {"question": "Who wrote Romeo and Juliet?", "correct_answer": "William Shakespeare"},
#         # Add more tasks...
#     ]

#     meta_search = MetaAgentSearch("gpt-3.5-turbo")
#     results = meta_search.search(tasks)

#     print(f"Best agent performance: {results['best_performance']}")
#     print("Best agent code:")
#     print(results['best_agent'].__class__.__name__)

#     # Save results
#     save_results(results, "meta_agent_search_results.json")

# if __name__ == "__main__":
#     main()
