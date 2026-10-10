import sys, json, pandas as pd
from vllm import LLM, SamplingParams

model, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
df = pd.read_csv("HarmBench/data/behavior_datasets/harmbench_behaviors_text_all.csv")
df = df[df.FunctionalCategory == "standard"].sample(n, random_state=0)

llm = LLM(model, dtype="bfloat16", max_model_len=4096,
          gpu_memory_utilization=0.11, enforce_eager=True)
tok = llm.get_tokenizer()
prompts = [tok.apply_chat_template([{"role": "user", "content": b}],
           tokenize=False, add_generation_prompt=True) for b in df.Behavior]
outs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=1024))

json.dump([{"id": i, "category": c, "behavior": b, "output": o.outputs[0].text}
           for i, c, b, o in zip(df.BehaviorID, df.SemanticCategory, df.Behavior, outs)],
          open(out, "w"), indent=1)
