import os
import json
import torch
import gradio as gr
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from tokenizers import Tokenizer
from pathlib import Path
from inference import Pratham

MODEL_ID = "oncody/Pratham-117M"
root = Path(snapshot_download(MODEL_ID, revision=os.environ.get("MODEL_REVISION", "main"),
    allow_patterns=["config.json","model.safetensors","tokenizer.json"]))
tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
model = Pratham(**json.loads((root / "config.json").read_text())).eval()
model.load_state_dict(load_file(str(root / "model.safetensors")))
torch.set_num_threads(min(4, os.cpu_count() or 1))
def format_prompt(message, history):
    turns = []
    for turn in history:
        content = turn.get("content", "")
        if isinstance(content, list):
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        if turn.get("role") in ("user","assistant") and isinstance(content, str):
            turns.append(turn["role"].capitalize()+": "+content)
    return "\n".join(turns+["User: "+message,"Assistant:"])
@torch.inference_mode()
def respond(message, history):
    prompt = format_prompt(message, history)
    ids = tokenizer.encode(prompt).ids[-model.t:]
    output = []
    for _ in range(64):
        scores = model(torch.tensor([ids[-model.t:]]))[0,-1]
        values, indices = (scores / .8).topk(30)
        token = indices[torch.multinomial(values.softmax(-1),1)].item()
        if token == tokenizer.token_to_id("<eod>"): break
        ids.append(token); output.append(token)
        if "\nUser:" in tokenizer.decode(output): break
    text = tokenizer.decode(output).split("\nUser:")[0].strip()
    return text or "[The model ended this reply immediately.]"
if __name__ == "__main__":
    gr.ChatInterface(respond,title="Pratham-117M",description="Experimental English model trained from scratch. Responses can be inaccurate or incoherent.").queue(default_concurrency_limit=1).launch()
