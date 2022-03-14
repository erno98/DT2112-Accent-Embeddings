import pandas as pd
import numpy as np
import wave as wv
from pydub import AudioSegment
import os


dsets_path = "./datasets"
subdirs = [os.path.join(dsets_path, item) for item in os.listdir(dsets_path)]
rec_path = []

for dir in subdirs:
    if dir == './datasets/accent_archive':
        rec_path.append(os.path.join(dir, "recordings/"))
    else:
        rec_path.append(dir)


archive = rec_path[0]
# for file in os.listdir(archive):
#     audSeg = AudioSegment.from_mp3(os.path.join(archive, file))
#     audSeg.export(os.path.join(archive, f"{file.split('.')[0]}.wav"), format="wav")
#     print(os.path.join(archive, f"{file.split('.')[0]}.wav"))

arch_df = pd.read_csv(os.path.join(subdirs[0], "speakers_all.csv"))

arch_df = arch_df[arch_df["file_missing?"] == False]
arch_df = arch_df[arch_df["native_language"] != "synthesized"]
arch_df = arch_df.drop(columns=["Unnamed: 9", "Unnamed: 10", "Unnamed: 11", "file_missing?"])
cols = arch_df.birthplace.str.split(", ", -1, expand=False)
arch_df.birthplace = cols.apply(lambda x: x[-1])
arch_df.filename = arch_df.filename.apply(lambda x: os.path.join(rec_path[0], x))
arch_df.filename = arch_df.filename.apply(lambda x: x + ".wav")

english_df = pd.read_excel("wikipedia_eng_lng_pop.ods", engine="odf")
english_df = english_df[["Country", "Total English speakers (%)"]]
english_df.Country = english_df.Country.str.lower()
english_df = english_df.rename(columns={"Country": "country", "Total English speakers (%)": "eng_speakers"})
print(english_df)
arch_df = pd.merge(left=arch_df, right=english_df, on="country")
arch_df["line"] = "Please call Stella. Ask her to bring these things with her from the store: Six spoons of fresh snow peas, five thick slabs of blue cheese, and maybe a snack for her brother Bob. We also need a small plastic snake and a big toy frog for the kids. She can scoop these things into three red bags, and we will go meet her Wednesday at the train station."
print(arch_df.columns)
arch_df["accent"] = arch_df["native_language"]
accents = {
    "usa": "american",
    "canada": "canadian",
    "uk": 'british',
    "australia": "australian",
    "ireland": "irish"
}

for country in accents:
    arch_df["accent"] = arch_df["accent"].where((arch_df["birthplace"] != country) | (arch_df["native_language"] != "english"), accents[country])

arch_df["accent_group"] = arch_df["accent"]
indian_languages = ["hindi", "bengali", "urdu", "marathi", "gujarati", "punjabi"]
south_slavic = ["croatian", "slovenian", "bulgarian"]
west_slavic = ["polish", "slovak"]
east_slavic = ["russian", "ukrainian"]
scandinavian = ["danish", "norwegian", "swedish"]
dutch = ["dutch", "vlaams", "afrikaans", "frisian"]
arch_df["accent_group"] = arch_df.accent_group.apply(lambda x: "indian" if x in indian_languages else x)
arch_df["accent_group"] = arch_df.accent_group.apply(lambda x: "dutch" if x in dutch else x)
arch_df["accent_group"] = arch_df.accent_group.apply(lambda x: "south slavic" if x in south_slavic else x)
arch_df["accent_group"] = arch_df.accent_group.apply(lambda x: "west slavic" if x in west_slavic else x)
arch_df["accent_group"] = arch_df.accent_group.apply(lambda x: "east slavic" if x in east_slavic else x)
arch_df["accent_group"] = arch_df.accent_group.apply(lambda x: "scandinavian" if x in scandinavian else x)

arch_df = arch_df.drop(1441)

final_df = arch_df

for path in rec_path[1:]:
    name = path.split("/")[-1]
    accent, _, gender = name.split("_")
    lndxdf = pd.read_csv(os.path.join(path, "line_index.csv"), header=None, names=["_", "filename", "line"])
    lndxdf.filename = lndxdf.filename.apply(lambda x: x + ".wav")
    lndxdf.filename = lndxdf.filename.apply(lambda x: os.path.join(path, x[1:]))
    lndxdf = lndxdf.drop(columns=['_'])
    lndxdf["age"] = np.NAN
    lndxdf["age_onset"] = 0.0
    lndxdf["birthplace"] = None
    lndxdf["native_language"] = "english"
    lndxdf["sex"] = gender
    lndxdf["speakerid"] = None
    lndxdf["country"] = "ireland" if accent=="irish" else "uk"
    lndxdf["eng_speakers"] = 98.37 if accent=="irish" else 97.74
    lndxdf["accent"] = accent
    lndxdf["accent_group"] = accent
    final_df = pd.concat([final_df, lndxdf], axis=0, ignore_index=True)

print(final_df[["country", "accent"]])
final_df["line"] = final_df["line"].apply(lambda x: x.replace(",", " "))
final_df.to_csv("dataset.csv")

