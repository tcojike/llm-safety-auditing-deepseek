import argparse, json, random, importlib.util
import pandas as pd
from vllm import LLM, SamplingParams

p = argparse.ArgumentParser()
p.add_argument("model")
p.add_argument("out")
p.add_argument("n_templates", nargs="?", type=int, default=5,
               help="jailbreaks per behavior (HarmBench random_subset_5 uses 5; -1 = all 114)")
p.add_argument("--seed", type=int, default=1)  # HarmBench HumanJailbreaks default
p.add_argument("--strip-think", action="store_true",
               help="score only text after </think> (raw output kept in generation_raw)")
args = p.parse_args()

# HumanJailbreaks templates are a Python list, not a file; load it directly so we
# don't trigger baselines/__init__.py (which imports every attack's dependencies).
spec = importlib.util.spec_from_file_location(
    "jailbreaks", "HarmBench/baselines/human_jailbreaks/jailbreaks.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
JAILBREAKS = list(mod.JAILBREAKS)

df = pd.read_csv("HarmBench/data/behavior_datasets/harmbench_behaviors_text_all.csv")

# Mirror HumanJailbreaks.generate_test_cases: seed once, shuffle the list in place
# for every behavior in the CSV, take the first k, prompt = f"{jb}\n\n{behavior}".
# We shuffle for non-standard rows too so the selection matches HarmBench's.
random.seed(args.seed)
rows = []
for _, b in df.iterrows():
    random.shuffle(JAILBREAKS)
    if b.FunctionalCategory != "standard":
        continue
    chosen = JAILBREAKS if args.n_templates == -1 else JAILBREAKS[:args.n_templates]
    for jb in chosen:
        rows.append({"id": b.BehaviorID, "template_idx": mod.JAILBREAKS.index(jb),
                     "prompt": f"{jb}\n\n{b.Behavior}"})

llm = LLM(args.model, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.85)
tok = llm.get_tokenizer()
fmt = [tok.apply_chat_template([{"role": "user", "content": r["prompt"]}],
       tokenize=False, add_generation_prompt=True) for r in rows]
outs = llm.generate(fmt, SamplingParams(temperature=0, max_tokens=4096))

res = {}
for r, o in zip(rows, outs):
    text = o.outputs[0].text
    gen = text.split("</think>", 1)[-1].strip() if args.strip_think else text
    res.setdefault(r["id"], []).append({
        "test_case": r["prompt"], "generation": gen, "generation_raw": text,
        "template_idx": r["template_idx"]})
json.dump(res, open(args.out, "w"), indent=1)
