import sys
from threading import Thread

import gradio as gr
import torch
from transformers import AutoModel, AutoProcessor
from transformers import StoppingCriteria, TextIteratorStreamer, StoppingCriteriaList
#device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = "cuda:0"
import psutil

def set_cpu_affinity(core_ids):
    p = psutil.Process()
    p.cpu_affinity(core_ids)

# Beschränken Sie den aktuellen Prozess auf die Verwendung der ersten 8 Kerne (0-7)
#set_cpu_affinity(range(16))

model = AutoModel.from_pretrained("unum-cloud/uform-gen2-qwen-500m", trust_remote_code=True).to(device)
processor = AutoProcessor.from_pretrained("unum-cloud/uform-gen2-qwen-500m", trust_remote_code=True)

class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [151645]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

@torch.no_grad()
def response(message, history, image):
    stop = StopOnTokens()

    messages = [{"role": "system", "content": "You are a helpful assistant."}]

    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})

    if len(messages) == 1:
        message = f" <image>{message}"

    messages.append({"role": "user", "content": message})

    model_inputs = processor.tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    )

    image = (
            processor.feature_extractor(image)
            .unsqueeze(0)
    )

    attention_mask = torch.ones(
        1, model_inputs.shape[1] + processor.num_image_latents - 1
    )
    
    model_inputs = {
        "input_ids": model_inputs,
        "images": image,
        "attention_mask": attention_mask
    }

    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

    streamer = TextIteratorStreamer(processor.tokenizer, timeout=30., skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        model_inputs,
        streamer=streamer,
        max_new_tokens=35,
        stopping_criteria=StoppingCriteriaList([stop])
        )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    history.append([message, ""])
    partial_response = ""
    for new_token in streamer:
        partial_response += new_token
        history[-1][1] = partial_response
        yield history, gr.Button(visible=False), gr.Button(visible=True, interactive=True)


with gr.Blocks() as demo:
    with gr.Row():
        image = gr.Image(type="pil")

        with gr.Column():
            chat = gr.Chatbot(show_label=False)
            message = gr.Textbox(interactive=True, show_label=False, container=False)

            with gr.Row():
                gr.ClearButton([chat, message])
                stop = gr.Button(value="Stop", variant="stop", visible=False)
                submit = gr.Button(value="Submit", variant="primary")

    with gr.Row():
        gr.Examples(
            [
                ["images/interior.jpg", "Describe the image accurately."],
                ["images/srew.png", "What is the image about?"],
                ["images/interior-1.jpg", "Describe the image accurately."],
            ],
            [image, message],
            label="Captioning"
        )

    response_handler = (
        response,
        [message, chat, image],
        [chat, submit, stop]
    )
    postresponse_handler = (
        lambda: (gr.Button(visible=False), gr.Button(visible=True)),
        None,
        [stop, submit]
    )

    event1 = message.submit(*response_handler)
    event1.then(*postresponse_handler)
    event2 = submit.click(*response_handler)
    event2.then(*postresponse_handler)

    stop.click(None, None, None, cancels=[event1, event2])

demo.queue()
demo.launch()