import sys, json, pandas as pd
from vllm import LLM, SamplingParams

model, out = sys.argv[1], sys.argv[2]
df = pd.read_csv("HarmBench/data/behavior_datasets/harmbench_behaviors_text_all.csv")
df = df[df.FunctionalCategory == "standard"]
llm = LLM(model, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.85)
tok = llm.get_tokenizer()
prompts = [tok.apply_chat_template([{"role": "user", "content": b}],
           tokenize=False, add_generation_prompt=True) for b in df.Behavior]
outs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=4096))
res = {bid: [{"test_case": b, "generation": o.outputs[0].text}]
       for bid, b, o in zip(df.BehaviorID, df.Behavior, outs)}
json.dump(res, open(out, "w"), indent=1)
