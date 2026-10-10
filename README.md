# BigData2026 Assignment 1 — Submission

This repository is my submission for BigData2026 Assignment 1. It contains the code and notebooks used to explore low-rank adaptation and the effect of different singular-value regions on LLM fine-tuning using the PiSSA approach.

The project focuses on analyzing how different components of the SVD spectrum affect model adaptation and evaluation quality on mathematical reasoning tasks.

## Files in this submission

- `finetune_q1-q3.ipynb` — main fine-tuning and evaluation workflow for Questions 1 to 3.
- `finetune_q4.ipynb` — additional analysis of the singular-value structure and low-rank behavior for Question 4.

## What this submission includes

- A modified PiSSA-based fine-tuning implementation for `meta-llama/Llama-3.2-1B`
- Training and evaluation pipelines for MetaMathQA, GSM8K, and MATH
- Experiment settings for different low-rank configurations
- A singular-value analysis section for Question 4

## Running on a local machine

I added code to make it possible to run these notebooks on a local machine as well.

If you want to run the code locally, you can simply:

1. Uncomment the local Hugging Face token login code.
2. Comment out the Kaggle-specific login code used in the Kaggle environment.

This is the key idea:

```python
# import os
# from getpass import getpass
# from huggingface_hub import login

# try:
#     # Optional: load from a .env file (pip install python-dotenv)
#     from dotenv import load_dotenv
#     load_dotenv()
# except ImportError:
#     pass

# try:
#     token = os.environ.get("HF_TOKEN") or getpass("Enter your Hugging Face token: ")
#     login(token=token)
#     print("Hugging Face login: OK")
# except Exception as exc:
#     print("HF login was not completed automatically:", exc)
#     print("If the model is already cached and accessible, you can continue.")
```

In other words, if the notebook is being run on a local machine, comment the Kaggle login block and uncomment the local login block. If it is being run on Kaggle, keep the Kaggle version enabled and use the Kaggle secret named `HF_TOKEN`.

## Recommended local setup

Before running the notebook locally, make sure the required Python packages are installed:

```bash
pip install torch transformers datasets accelerate peft evaluate matplotlib numpy
```

Also make sure you have access to the gated model `meta-llama/Llama-3.2-1B` on Hugging Face and that your token is available in the environment or entered manually. For local execution, create a `.env` file in the project folder and include:

```env
HF_TOKEN=your_huggingface_token_here
```

The notebook is set up to load the token from `.env` automatically if `python-dotenv` is installed.

## Notes for this assignment

This repository is organized for assignment submission and demonstration purposes. The notebooks are designed to run in either a Kaggle environment or a local development environment, depending on the setup used by the user.


