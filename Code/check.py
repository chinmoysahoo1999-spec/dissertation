import json

with open("hallucination_10k.json") as f:
    data = json.load(f)

preview = data[:4]  # take first 3 samples

with open("preview.json", "w") as f:
    json.dump(preview, f, indent=2)

print("preview.json created")
