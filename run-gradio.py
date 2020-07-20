import gradio
from gpt import get_model


model_small, tokenizer_small = get_model("gpt2")
model_xl, tokenizer_xl = get_model("gpt2-xl")

def predict(inp, model_type):
    if model_type == "gpt2-xl":
        model, tokenizer = model_xl, tokenizer_xl
    else:
        model, tokenizer = model_small, tokenizer_small
    input_ids = tokenizer.encode(inp, return_tensors='tf')
    beam_output = model.generate(input_ids, max_length=100, num_beams=5,
                                 no_repeat_ngram_size=2, early_stopping=True)
    output = tokenizer.decode(beam_output[0], skip_special_tokens=True,
                              clean_up_tokenization_spaces=True)
    return ".".join(output.split(".")[:-1]) + "."

 

INPUTS = [gradio.inputs.Textbox(lines=2, label="Input Text"),
            gradio.inputs.Radio(choices=["gpt2-xl", "gpt2-small"], label="Choose "
                                                                "between xl "
                                                                "and small")]
OUTPUTS = gradio.outputs.Textbox()
examples = [
    ["The British government has secured early access to more than 90 "
     "million vaccine doses.", "gpt2-xl"],
    ["The three hardest things about software engineering are", "gpt2-xl"]
]
INTERFACE = gradio.Interface(fn=predict, inputs=INPUTS, outputs=OUTPUTS, title="GPT-2",
                 description="GPT-2 is a large transformer-based language "
                             "model with 1.5 billion parameters, trained on "
                             "a dataset[1] of 8 million web pages. GPT-2 is "
                             "trained with a simple objective: predict the "
                             "next word, given all of the previous words "
                             "within some text. You can configre small vs xl below (the xl model takes longer to run but generates better text.",
                 thumbnail="https://github.com/gradio-app/gpt-2/raw/master/screenshots/interface.png?raw=true",
                             examples=examples,
                 capture_session=False)

INTERFACE.launch(inbrowser=True)
