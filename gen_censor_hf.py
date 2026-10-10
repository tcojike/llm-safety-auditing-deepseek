import sys, json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id, out = sys.argv[1], sys.argv[2]
prompts = [l.strip() for l in open("censor_prompts.txt") if l.strip()]
tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to("cuda")

res = []
for i in range(0, len(prompts), 4):
    batch = prompts[i:i+4]
    texts = [tok.apply_chat_template([{"role": "user", "content": p}],
             tokenize=False, add_generation_prompt=True) for p in batch]
    enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        gen = model.generate(**enc, max_new_tokens=1536, do_sample=False)
    for p, g in zip(batch, gen):
        res.append({"prompt": p, "output": tok.decode(g[enc.input_ids.shape[1]:], skip_special_tokens=False)
                    .replace(tok.eos_token or "", "")})
json.dump(res, open(out, "w"), indent=1)
