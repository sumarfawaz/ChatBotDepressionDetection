import pandas as pd
import json
import uuid

# Read the CSV file
df = pd.read_csv('D:/EmoAssist/reddit_text-davinci-002.csv')  # Replace 'your_large_file.csv' with the actual file path

# Create a list to store intents
intents = []

# Iterate over rows in the DataFrame
for index, row in df.iterrows():
    # Assuming 'Prompt' and 'Response' are the column names in your CSV
    pattern = str(row['prompt'])
    response = str(row['completion'])

    # Generate a random unique ID for the intent tag
    intent_tag = str(uuid.uuid4())

    # Create an intent entry
    intent_entry = {
        "tag": intent_tag,
        "patterns": [pattern],
        "responses": [response]
    }

    # Append the intent entry to the list
    intents.append(intent_entry)

# Save as JSON
with open('intents.json', 'w') as json_file:
    json.dump(intents, json_file, indent=4)
