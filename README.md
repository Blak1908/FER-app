# Vietnamese AI Chatbot with Transformer

This is an AI chatbot that uses a transformer model as its main model. The chatbot can interact with users in a variety of ways, including answering questions, providing recommendations, and more. The chatbot can also analyze users based on their age, gender, interests, and more. Additionally, the chatbot has integration with speech-to-text and text-to-speech models.


## Project Overview

The goal of this project is to create an intelligent and interactive chatbot capable of understanding and generating human-like responses in Vietnamese. We utilize the transformer model, a state-of-the-art deep learning architecture in the field of NLP, to achieve this objective. The chatbot is trained on a diverse dataset to ensure accurate and contextually relevant responses.

![Chatbot using Transformer Architecture](https://github.com/blak-tran/AI-Chatbot-Synthesis/blob/da19064f92e8aa2da7d6dfacc4bf236ac38a18fb/assets/transformer_architect.png)


## Features

User interaction
Multi-task handling, including answering questions, providing recommendations, and more
Context-aware responses, resulting in more natural interactions
User analysis
Integration with speech-to-text and text-to-speech models

## Technologies Used

- **Python:** The core programming language used for building the chatbot.
- **Tensorflow:** Deep learning framework utilized for implementing the transformer model.
- **Transformers Library:** Leveraged for accessing pre-trained transformer models.
- **FastAPI:** Web framework used for building the chatbot's user interface.

## Getting Started

### Usage
To use the chatbot, follow these steps:
1. **Setup**
 We used Python 3.9.9 and PyTorch 1.10.1 to train and test our models, but the codebase is expected to be compatible with Python 3.8-3.11 and recent PyTorch versions. The codebase also depends on a few Python packages, most notably OpenAI's tiktoken for their fast tokenizer implementation.
pip install -r requirement.txt

2. **Inference**
   ```bash
   python main_flow_inference.py 
