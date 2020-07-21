import gradio
from gpt import get_model


model_small, tokenizer_small = get_model("gpt2")
model_large, tokenizer_large = get_model("gpt2-large")

def predict(inp, model_type):
    if model_type == "gpt2-large":
        model, tokenizer = model_large, tokenizer_large
        input_ids = tokenizer.encode(inp, return_tensors='tf')
        beam_output = model.generate(input_ids, max_length=42, num_beams=5,
                                     no_repeat_ngram_size=2,
                                     early_stopping=True)
        output = tokenizer.decode(beam_output[0], skip_special_tokens=True,
                                  clean_up_tokenization_spaces=True)
    else:
        model, tokenizer = model_small, tokenizer_small
        input_ids = tokenizer.encode(inp, return_tensors='tf')
        beam_output = model.generate(input_ids, max_length=70, num_beams=5,
                                 no_repeat_ngram_size=2, early_stopping=True)
        output = tokenizer.decode(beam_output[0], skip_special_tokens=True,
                              clean_up_tokenization_spaces=True)
    if output.count(".") >= 2:
        output = ".".join(output.split(".")[:-1]) + "."
    return output

 

INPUTS = [gradio.inputs.Textbox(lines=2, label="Input Text"),
            gradio.inputs.Radio(choices=["gpt2-small", "gpt2-large"],
                                label="Choose "
                                                                "between "
                                                                         "small "
                                                                "and large")]
OUTPUTS = gradio.outputs.Textbox()
examples = [
    ["The toughest thing about software engineering is", "gpt2-large"],
    ["The future of AI ", "gpt2-large"],
    ["Is this the real life? Is this just fantasy?", "gpt2-small"]
]
INTERFACE = gradio.Interface(fn=predict, inputs=INPUTS, outputs=OUTPUTS, title="GPT-2",
                 description="GPT-2 is a large transformer-based language "
                             "model with 1.5 billion parameters, trained on "
                             "a dataset of 8 million web pages. GPT-2 is "
                             "trained with a simple objective: predict the "
                             "next word, given all of the previous words "
                             "within some text. You can configure small vs "
                             "large below: the large model takes longer to "
                             "run ("
                             "55s vs 30s) "
                             "but generates better text.",
                 thumbnail="https://github.com/gradio-app/gpt-2/raw/master/screenshots/interface.png?raw=true",
                             examples=examples,
                 capture_session=False)

INTERFACE.launch(inbrowser=True)
