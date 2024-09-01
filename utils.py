import json
import os
from typing import List, Dict, Any
import requests
# from requests.structures import CaseInsensitiveDict as _HeadersMapping
import matplotlib.pyplot as plt
# import ollama
import anthropic
from anthropic.types import ContentBlock
import logging

logger = logging.getLogger(__name__)


def load_data(dataset_name: str) -> List[Dict[str, Any]]:
    file_path = f"data/{dataset_name}.json"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    with open(file_path, 'r') as f:
        data = json.load(f)

    return data

def query_claude(prompt: str) -> str:
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    try:
        message = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            temperature=0.5,
            system="You are an AI assistant designed to generate and improve AI agents.",
            messages=[{"role": "user", "content": prompt}]
        )
        if not message.content:
            raise ValueError("Empty response from Claude")

        # Extract text from ContentBlock objects
        response_text = ""
        for block in message.content:
            if isinstance(block, anthropic.types.TextBlock):
                response_text += block.text

        if not response_text:
            raise ValueError("No text content in Claude's response")

        return response_text
    except Exception as e:
        logger.error(f"Error querying Claude: {e}")
        return ""


def query_ollama(prompt, model="llama3"):
    logger.debug(f"Querying Ollama with prompt: {prompt[:100]}...")
    try:
        response = requests.post('http://localhost:11434/api/generate', json={
            "model": model,
            "prompt": prompt
        })
        response.raise_for_status()

        # Print raw response for debugging
        logger.debug(f"Raw Ollama response: {response.text}")

        # Manual parsing of the response
        lines = response.text.split('\n')
        full_response = ""
        for line in lines:
            try:
                data = json.loads(line)
                if 'response' in data:
                    full_response += data['response']
            except json.JSONDecodeError:
                continue

        if full_response:
            logger.debug(f"Ollama response received: {full_response[:100]}...")
            return full_response.strip()
        else:
            logger.warning("Failed to find a valid response from Ollama.")
            return "Error: No valid response from Ollama"
    except requests.RequestException as e:
        logger.error(f"Error querying Ollama: {e}")
        return f"Error: Failed to query Ollama - {str(e)}"


# def query_language_model(model: str, prompt: str) -> str:
#     if model.startswith("anthropic/"):
#         # Use Anthropic API for meta-agent
#         import anthropic
#         client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
#         response = client.completions.create(
#             model=model.split("/")[1],
#             prompt=prompt,
#             max_tokens_to_sample=100
#         )
#         return response.completion.strip()
#     else:
#         # Use Ollama API at localhost:11434
#         return "user ollama"


def evaluate_performance(agent_results: List[Dict], baseline_results: List[List[Dict]]):
    def calculate_accuracy(results):
        correct = sum(1 for r in results if r['answer'].strip() == r['task']['answer'].strip())
        try:
            if len(results) > 0:
                return correct/results
        except:
            return 0  # Return 0 if division by zero would occur

    agent_accuracy = calculate_accuracy(agent_results)
    baseline_accuracies = [calculate_accuracy(baseline) for baseline in baseline_results]

    # Plot results
    # plt.figure(figsize=(10, 6))
    # plt.bar(['Meta Agent'] + [f'Baseline {i+1}' for i in range(len(baseline_accuracies))],
    #         [agent_accuracy] + baseline_accuracies)
    # plt.title('Performance Comparison')
    # plt.ylabel('Accuracy')
    # plt.savefig('performance_comparison.png')
    # plt.close()

    return {
        'meta_agent_accuracy': agent_accuracy,
        'baseline_accuracies': baseline_accuracies
    }


def save_results(results: Dict[str, Any], filename: str):
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)

def load_results(filename: str) -> Dict[str, Any]:
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Results file not found: {filename}")

    with open(filename, 'r') as f:
        results = json.load(f)

    return results
