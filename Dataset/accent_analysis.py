import pandas as pd

dataset = pd.read_csv("dataset.csv")

for file in dataset["filename"]:
    try:
        f = open(file, "r")
        f.close()
    except Exception as e:
        print(e)


accent_counts = pd.DataFrame(dataset["accent"].value_counts())
accent_counts = accent_counts.rename(columns={"accent": "count"})
accent_counts.index.name = "accent"
accent_counts.to_csv("accent_counts.csv")

accent_counts = pd.DataFrame(dataset["accent_group"].value_counts())
accent_counts = accent_counts.rename(columns={"accent": "count"})
accent_counts.index.name = "accent_group"
accent_counts.to_csv("accent_group_counts.csv")

