import os
import argparse
import pandas as pd
from llama_cpp import Llama
from tqdm.auto import tqdm
import time

# Constants for column names
SENT_COL = "llm_sentiment"
TEXT_COL = "clean_body"

# Prompt structure
PREFIX = (
    "<|system|>\n"
    "You are a sentiment analysis assistant specialized in political and geopolitical forum discussions. "
    "Users post multi-sentence comments, often with nuanced opinions or criticism. "
    "Analyze the overall tone of each comment and classify it as exactly one of: "
    "Positive, Negative, or Neutral. "
    "Respond ONLY with that single label and nothing else.\n"
    "</s>\n"

    "<|user|>\n"
    "Text: \"While I appreciate the intent behind the new policy, its rollout was chaotic and left many communities worse off.\"\n"
    "</s>\n"
    "<|assistant|>\n"
    "Negative\n\n"

    "<|user|>\n"
    "Text: \"The committee’s report lays out both strengths and weaknesses without taking a clear stance.\"\n"
    "</s>\n"
    "<|assistant|>\n"
    "Neutral\n\n"

    "<|user|>\n"
    "Text: \"Despite fierce opposition, the strategy has already shown promising signs of stabilizing the region.\"\n"
    "</s>\n"
    "<|assistant|>\n"
    "Positive\n\n"

    "<|user|>\n"
    f"Text: \""
    )
SUFFIX = "\"\n</s>\n<|assistant|>\n"

GEN_TOKS = 3

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to input CSV')
    parser.add_argument('--output', type=str, required=True, help='Path to output CSV')
    parser.add_argument('--model-path', type=str, default='models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf')
    parser.add_argument("--n-threads", type=int, default=4, help="Number of threads")
    parser.add_argument(
        "--n-ctx", type=int, default=512, help="Max context window (tokens)"
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=5000,
        help="Checkpoint every N rows",
    )
    args = parser.parse_args()
    return args

def truncate_to_max_tokens(text: str, llm: Llama, max_tokens: int) -> str:
    # Truncate text to the first max_tokens based on tokenization
    btext = text.encode('utf-8')
    tok_ids = llm.tokenize(btext)
    if len(tok_ids) <= max_tokens:
        return text
    detok = llm.detokenize(tok_ids[:max_tokens])
    if isinstance(detok, (bytes, bytearray)):
        detok = detok.decode('utf-8', errors='ignore')
    return detok

def main():
# Get arguments
    args = parse_args()

    # Initialize the Llama model
    llm = Llama(
        model_path=args.model_path,
        n_threads=args.n_threads,
        n_ctx=args.n_ctx,
        verbose=False,
    )
    print("Using model:", args.model_path)

    # Load the input CSV file
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file {args.input} does not exist.")
    # Load data
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows from {args.input}")
    df[SENT_COL] = None

    # Pre-truncate texts
    prefix_toks = len(llm.tokenize(PREFIX.encode('utf-8')))
    suffix_toks = len(llm.tokenize(SUFFIX.encode('utf-8')))
    max_input_tokens = args.n_ctx - (prefix_toks + suffix_toks + GEN_TOKS)
    print(f"Max input tokens available: {max_input_tokens}")
    if max_input_tokens <= 0:
        raise RuntimeError("Prompt overhead too large for your n_ctx!")
    
    t0 = time.perf_counter()
    try:
        for idx, raw in tqdm(df[TEXT_COL].fillna("").items(), total=len(df)):
        
            # Truncate each text to the max input tokens
            txt = truncate_to_max_tokens(raw, llm, max_input_tokens)
            prompt = PREFIX + txt + SUFFIX
            # print(prompt)
            out = llm(prompt, max_tokens=GEN_TOKS, stop=["\n"]).get("choices")[0].get("text").strip()
            if out not in {"Positive","Negative","Neutral"}:
                # Check if it starts with a label
                if out.startswith("Pos"):
                    out = "Positive"
                elif out.startswith("Neg"):
                    out = "Negative"
                elif out.startswith("Neu"):
                    out = "Neutral"
                # log or fallback
                else:
                    print(f"[Warning] bad label `{out}` @ row {idx}")
                    out = "Neutral"  # or whatever default
            df.at[idx, SENT_COL] = out

            # Checkpointing
            if (idx > 0 and idx % args.checkpoint_every == 0):
                print(f"Checkpointing at row {idx}...")
                fname = f"{args.output}-checkpoint-{idx}.csv"
                df.iloc[:idx+1].to_csv(fname, index=False)
        
    except KeyboardInterrupt:
        print("Interrupted! Saving progress...")
        df.iloc[:idx+1].to_csv(args.output, index=False)
    
    # Final save
    df.to_csv(args.output, index=False)
    print(f"Annotation complete. Results saved to {args.output}")
    t1 = time.perf_counter()
    print(f"Annotated 100 rows in {t1-t0:.1f}s → "f"{len(df)/(t1-t0):.1f} rows/sec")
    
if __name__ == "__main__":
    main()