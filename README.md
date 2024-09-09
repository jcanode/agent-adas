# Automated Design of Agentic Systems (ADAS)

This project implements a meta-learning system for automatically designing and optimizing AI agents. It's based on the paper "Automated Design of Agentic Systems" and aims to create agents that can adapt quickly to new tasks.

## Project Structure

- `main.py`: Main script to run experiments
- `meta_agent_search.py`: Implements the Meta Agent Search algorithm
- `agent_framework.py`: Defines the base agent structure
- `baselines.py`: Implements baseline agent types
- `utils.py`: Utility functions for data loading, model querying, etc.
- `evaluator.py`: Implements the evaluation system for agents

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/jcanode/ADAS.git
   cd ADAS
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up your environment variables:
   ```
   export ANTHROPIC_API_KEY=your_api_key_here
   ```

## Usage

To run the main experiment:

```
python main.py
```

This will:
1. Load the sample tasks
2. Run the Meta Agent Search to discover new agent designs
3. Evaluate the discovered agents against baseline agents
4. Save the results to `experiment_results.json`

## Customization

- To add new baseline agents, modify the `baselines.py` file.
- To change the evaluation criteria, update the `evaluator.py` file.
- To use different tasks, update the data loading in `utils.py` and `main.py`.

## Results

The experiment results are saved in `experiment_results.json`. This file includes:
- The best agent discovered by Meta Agent Search
- Performance comparisons between the discovered agent and baseline agents
- Detailed evaluations of each agent's performance

## Contributing

Contributions to this project are welcome! Please fork the repository and submit a pull request with your changes.
