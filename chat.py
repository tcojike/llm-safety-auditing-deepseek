import sys, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = sys.argv[1]
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to("cuda")
history = []
print("Type a message. 'quit' exits, 'reset' clears history.")
while True:
    q = input("\nYou: ").strip()
    if q == "quit": break
    if q == "reset": history = []; continue
    history.append({"role": "user", "content": q})
    enc = tok.apply_chat_template(history, add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True).to("cuda")
    out = model.generate(**enc, max_new_tokens=1024, do_sample=False)
    reply = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=False)
    reply = reply.replace(tok.eos_token or "", "")
    print("\nModel:", reply)
    history.append({"role": "assistant", "content": reply})
