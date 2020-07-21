# GPT-2 on Gradio

This repo creates the interface on [Gradio Hub](https://hub.gradio.app). Credit to the awesome [transformers library](https://github.com/huggingface/transformers) and [OpenAI](https://github.com/openai/gpt-2).

![alt text](https://github.com/gradio-app/gpt-2/blob/master/screenshots/interface.png?raw=true)

- Loads `gpt-2`, but can load `gpt2-medium`, `gpt2-large` or `gpt2-xl`.
- `max_length` set to 100 (The max length of the sequence to be generated.)
- running `".".join(output.split(".")[:-1]) + "."` so that `max_length` doesn't stop an output abruptly. 
