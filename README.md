# LLM-Paper-To-Code

A collection of Jupyter notebooks implementing advanced LLM prompting techniques to solve Logic Grid Puzzles (Zebra Puzzles) using the ZebraLogicBench dataset.

## Overview

This repository demonstrates practical implementations of research papers on Large Language Model (LLM) prompting strategies. The notebooks use GitHub Models (free tier) to solve constraint satisfaction problems in the form of Logic Grid Puzzles.

Logic Grid Puzzles, also known as Zebra Puzzles, are constraint satisfaction problems where you must deduce a unique correct assignment of values to houses based on given clues. These puzzles are commonly used to test logical reasoning abilities in exams such as the Law School Admission Test (LSAT).

## Notebooks

### 1. Self_Consistency_Code.ipynb
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pacozaa/LLM-Paper-To-Code/blob/main/Self_Consistency_Code.ipynb)

Implements the **Self-Consistency** prompting technique, which improves reasoning accuracy by:
- Sampling multiple reasoning paths from the LLM
- Extracting final answers from each sample
- Selecting the most consistent (most common) answer

This technique is based on the paper: "Self-Consistency Improves Chain of Thought Reasoning in Language Models"

**Key Features:**
- Uses Chain-of-Thought (CoT) prompting
- Generates multiple samples with temperature-based sampling
- Implements majority voting for answer selection
- Compares results against ground truth

### 2. DynamicCheatSheet.ipynb
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pacozaa/LLM-Paper-To-Code/blob/main/DynamicCheatSheet.ipynb)

Implements a dynamic few-shot learning approach with iteratively generated "cheat sheets":
- Dynamically generates example-based guidance
- Uses previously solved examples to improve performance
- Demonstrates adaptive prompting strategies

This technique is based on the paper: "Dynamic Cheatsheet: Test-Time Learning with Adaptive Memory" by Suzgun et al.

## Prerequisites

- Python 3.7+
- Jupyter Notebook or Google Colab
- GitHub account (for GitHub Models free tier)

## Setup

1. **Install dependencies:**
   ```bash
   pip install azure-ai-inference datasets
   ```

2. **Get a GitHub token:**
   - GitHub Models provides free access to various LLMs
   - You'll need a GitHub personal access token
   - Check rate limits: [GitHub Models Documentation](https://docs.github.com/en/github-models/use-github-models/prototyping-with-ai-models#rate-limits)

3. **Set up Hugging Face access (for dataset):**
   - You'll need access to the ZebraLogicBench dataset
   - Login to Hugging Face in the notebook

## Usage

### Running in Google Colab (Recommended)
1. Click on the "Open In Colab" badge above each notebook
2. Follow the setup instructions in the notebook
3. Add your GitHub token when prompted
4. Run all cells

### Running Locally
1. Clone this repository:
   ```bash
   git clone https://github.com/pacozaa/LLM-Paper-To-Code.git
   cd LLM-Paper-To-Code
   ```

2. Start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

3. Open the desired notebook and follow the instructions

## Dataset

This project uses the **ZebraLogicBench** dataset:
- **Dataset Viewer:** [Hugging Face Dataset](https://huggingface.co/datasets/allenai/ZebraLogicBench-private/viewer/grid_mode/test)
- **Blog Post:** [Zebra Logic Benchmark](https://huggingface.co/blog/yuchenlin/zebra-logic)
- The dataset contains logic grid puzzles of various sizes (2x2, 3x3, etc.)

## Prompt Templates

The repository includes YAML prompt templates for few-shot learning:
- `zebra-logic-1.prompt.yml` - Basic prompt template
- `zebra-logic-2-longer.prompt.yml` - Extended prompt template with more examples

These templates demonstrate:
- System prompts for puzzle-solving
- Few-shot examples with reasoning steps
- Structured JSON output format

## Techniques Implemented

1. **Chain-of-Thought (CoT) Prompting:** Encouraging step-by-step reasoning
2. **Self-Consistency:** Sampling multiple reasoning paths and majority voting
3. **Few-Shot Learning:** Using example problems to guide the model
4. **Dynamic Examples:** Iteratively building better prompts from solved examples

## Models Tested

The notebooks work with various models available through GitHub Models:
- GPT-4 variants: `openai/gpt-4o`, `openai/gpt-4.1`
- Microsoft Phi: `phi-4` (may also be referenced as `microsoft/phi-4`)

Note: Model identifiers should match the GitHub Models API naming conventions. Some models may accept shorthand names.

## References

- **Self-Consistency Paper:** Wang et al. "Self-Consistency Improves Chain of Thought Reasoning in Language Models"
- **Dynamic Cheatsheet Paper:** Mirac Suzgun, Mert Yuksekgonul, Federico Bianchi, Dan Jurafsky, James Zou. "Dynamic Cheatsheet: Test-Time Learning with Adaptive Memory" - [arXiv:2504.07952v1](https://arxiv.org/abs/2504.07952v1#S4)
- **ZebraLogicBench:** A benchmark for evaluating logical reasoning in language models
- **GitHub Models:** [Prototyping with AI Models](https://docs.github.com/en/github-models)

## Contributing

Contributions are welcome! Feel free to:
- Add new prompting techniques
- Improve existing implementations
- Add more comprehensive examples
- Enhance documentation

## License

This project is open source and available for educational purposes.
