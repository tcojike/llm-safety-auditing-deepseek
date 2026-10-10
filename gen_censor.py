import sys, json
from vllm import LLM, SamplingParams

model, out = sys.argv[1], sys.argv[2]
prompts = [l.strip() for l in open("censor_prompts.txt") if l.strip()]
llm = LLM(model, dtype="bfloat16", max_model_len=4096,
          gpu_memory_utilization=0.11, enforce_eager=True)
tok = llm.get_tokenizer()
fmt = [tok.apply_chat_template([{"role": "user", "content": p}],
       tokenize=False, add_generation_prompt=True) for p in prompts]
outs = llm.generate(fmt, SamplingParams(temperature=0, max_tokens=1536))
json.dump([{"prompt": p, "output": o.outputs[0].text} for p, o in zip(prompts, outs)],
          open(out, "w"), indent=1)
