import re
import pandas as pd

# Load the dataset
df = pd.read_csv("data/gold_standard.csv", usecols=["id", "weak_label", "clean_body"])

# Define the text column
TEXT_COL = "clean_body"

# Define a regex pattern to identify bot messages
bot_pattern = re.compile(
    r"\bi am a bot\b|this action was performed automatically|automatically removed",
    flags=re.IGNORECASE,
)

# build a boolean mask of “bot” rows
is_bot = df[TEXT_COL].fillna("").str.contains(bot_pattern)

# optional: inspect or save them
bots = df[is_bot]
print(f"Dropping {len(bots)} bot messages")
bots.to_csv("data/bot_messages.csv", index=False)

# keep only non-bot rows
df = df[~is_bot].reset_index(drop=True)

# Save the cleaned dataset
df.to_csv("data/cleaned_gold_standard.csv", index=False)