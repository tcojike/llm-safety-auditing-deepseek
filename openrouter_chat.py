import os, sys, argparse
from openai import OpenAI

p = argparse.ArgumentParser()
p.add_argument("model", help="OpenRouter model id, e.g. deepseek/deepseek-r1-distill-qwen-7b")
p.add_argument("prompt", nargs="?", help="one-shot prompt; omit for interactive chat")
p.add_argument("--temperature", type=float, default=0)
p.add_argument("--max-tokens", type=int, default=1024)
args = p.parse_args()

# Key comes from the environment, or from a gitignored .env file next to this script.
env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if "OPENROUTER_API_KEY" not in os.environ and os.path.exists(env_file):
    for line in open(env_file):
        k, _, v = line.strip().partition("=")
        if k == "OPENROUTER_API_KEY":
            os.environ[k] = v.strip().strip("'\"")
key = os.environ.get("OPENROUTER_API_KEY")
if not key or key == "paste-your-key-here":
    sys.exit(f"Put your OpenRouter key in {env_file} (create one at https://openrouter.ai/keys)")

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key,
                default_headers={"X-Title": "llm-safety-audit"})

def ask(messages):
    r = client.chat.completions.create(model=args.model, messages=messages,
                                       temperature=args.temperature,
                                       max_tokens=args.max_tokens)
    return r.choices[0].message.content or ""

if args.prompt:
    print(ask([{"role": "user", "content": args.prompt}]))
    sys.exit()

history = []
print("Type a message. 'quit' exits, 'reset' clears history.")
while True:
    q = input("\nYou: ").strip()
    if q == "quit": break
    if q == "reset": history = []; continue
    history.append({"role": "user", "content": q})
    reply = ask(history)
    print("\nModel:", reply)
    history.append({"role": "assistant", "content": reply})
