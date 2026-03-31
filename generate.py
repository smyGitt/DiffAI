import sys
import torch
import tiktoken

from config import ModelConfig, TrainConfig
from data import format_prompt_only, MSG_END
from model import DiffGPT, load_pretrained_gpt2


def generate_description(model, enc, diff_text, device="cpu",
                         max_new_tokens=80, temperature=0.7, top_k=50,
                         use_pretrained=False):
    model.eval()
    prompt = format_prompt_only(diff_text)
    prompt_ids = enc.encode(prompt)
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    if use_pretrained:
        out = model.generate(
            input_ids=idx,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            pad_token_id=enc.eot_token,
        )
        generated = enc.decode(out[0].tolist())
    else:
        out = model.generate(idx, max_new_tokens=max_new_tokens,
                             temperature=temperature, top_k=top_k)
        generated = enc.decode(out[0].tolist())

    marker = "<|msg|>\n"
    if marker in generated:
        response = generated.split(marker, 1)[1]
    else:
        response = generated[len(prompt):]

    end_marker = MSG_END
    if end_marker in response:
        response = response.split(end_marker, 1)[0]

    eot = "<|endoftext|>"
    if eot in response:
        response = response.split(eot, 1)[0]

    return response.strip()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mcfg = ModelConfig()
    tcfg = TrainConfig()
    enc = tiktoken.get_encoding("gpt2")

    checkpoint = tcfg.checkpoint_dir / "diffai_final.pt"

    if tcfg.use_pretrained:
        print(f"Loading pretrained {tcfg.pretrained_model}...")
        model = load_pretrained_gpt2(mcfg, tcfg.pretrained_model).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        use_pretrained = True
    else:
        print("Loading DiffGPT...")
        model = DiffGPT(mcfg).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        use_pretrained = False

    print("Model loaded. Enter a diff (paste then type END on a new line):\n")

    while True:
        lines = []
        try:
            for line in sys.stdin:
                line = line.rstrip("\n")
                if line.strip() == "END":
                    break
                lines.append(line)
        except EOFError:
            break

        if not lines:
            break

        diff_text = "\n".join(lines)
        desc = generate_description(model, enc, diff_text, device=device,
                                    use_pretrained=use_pretrained)
        print(f"\nGenerated description:\n{desc}\n")
        print("Enter another diff (or Ctrl+C to exit):\n")


if __name__ == "__main__":
    main()
