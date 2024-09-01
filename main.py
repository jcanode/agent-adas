# main.py
import logging
import argparse
from utils import load_data, evaluate_performance, save_results, query_ollama, query_claude
from agent_framework import BaseAgent
from meta_agent_search import MetaAgentSearch
from baselines import ChainOfThought, SelfRefine,  get_baseline_agents
from evaluation import Evaluator

from typing import List, Dict, Any
import os
# logger = logging.getLogger(__name__)

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_logging():
    log_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(log_dir, 'local_agents.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename=log_file,
        filemode='a'
    )

class ClaudeMetaAgent:
    def __init__(self):
        self.model = "claude-3-sonnet-20240229"

    def generate_agent(self, prompt: str) -> str:
        return query_claude(prompt)



class OllamaAgent:
    def __init__(self):
        self.model = "llama3"

    def forward(self, task: Dict[str, Any]) -> str:
        prompt = f"Passage: {task['passage']}\nQuestion: {task['question']}\nAnswer:"
        return query_ollama(prompt)

    def run(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [{"task": task, "answer": self.forward(task)} for task in tasks]




def parse_arguments():
    # Set up argument parser for different experiments and configurations
    pass

# def run_experiment(experiment_type, dataset, model):
#     # Load data
#     data = load_data(dataset)

#     # Initialize agents
#     meta_agent = MetaAgentSearch(model)
#     baseline_agents = [ChainOfThought(model), SelfRefine(model), LLMDebate(model)]

#     # Run Meta Agent Search
#     meta_agent_results = meta_agent.search(data)

#     # Run baselines
#     baseline_results = [agent.run(data) for agent in baseline_agents]

#     # Evaluate and compare results
    # evaluate_performance(meta_agent_results, baseline_results)


def main():
    # Sample tasks (replace with actual tasks from your dataset)
    # tasks = [
    #     {"question": "What is 2 + 2?", "correct_answer": "4"},
    #     {"question": "Who wrote Romeo and Juliet?", "correct_answer": "William Shakespeare"},
    #     # Add more tasks...
    # ]
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting the experiment")
    # setup_logging()
    evaluator = Evaluator(use_claude=True)

    tasks = load_data("sample_drop_data")
    logger.info(f"Loaded {len(tasks)} tasks from the dataset")

    # Run baseline agents using Ollama
    baseline_agents = get_baseline_agents()
    baseline_results = [agent.run(tasks) for agent in baseline_agents]
    logger.info(f"getting baseline performace. {baseline_results}")
    logger.info("Evaluating baseline agents")
    baseline_evaluations = []
    for agent, results in zip(baseline_agents, baseline_results):
        agent_evaluations = []
        for task, result in zip(tasks, results):
            scores = evaluator.evaluate_solution(task, result['answer'])
            evaluation_result = evaluator.format_results(task, result['answer'], scores)
            agent_evaluations.append(evaluation_result)
        baseline_evaluations.append(agent_evaluations)

    logger.info(f"baseline performance: {baseline_evaluations}")

    # Set up the meta agent using Claude
    meta_agent = ClaudeMetaAgent()
    logger.info("Set up Claude Meta Agent")


    # Run Meta Agent Search
    meta_search = MetaAgentSearch(meta_agent)
    logger.info("Starting Meta Agent Search")

    meta_results = meta_search.search(tasks)
    print(f"Best Meta Agent performance: {meta_results['best_performance']}")
    print("Best Meta Agent code:")
    print(meta_results['best_agent'])



    # Evaluate performance
    performance_comparison = evaluate_performance(meta_results['best_agent_results'], baseline_results)
    logger.info("Evaluating the best meta agent")
    best_agent_evaluations = []
    for task, result in zip(tasks, meta_results['best_agent_results']):
        scores = evaluator.evaluate_solution(task, result['answer'])
        evaluation_result = evaluator.format_results(task, result['answer'], scores)
        best_agent_evaluations.append(evaluation_result)

    # Combine all results
    final_results = {
        "meta_agent_results": meta_results,
        "baseline_results": baseline_results,
        "performance_comparison": performance_comparison,
        "best_agent_evaluations": best_agent_evaluations,
        "baseline_evaluations": baseline_evaluations
    }

    # Save results
    save_results(final_results, "experiment_results.json")



if __name__ == "__main__":
    main()
