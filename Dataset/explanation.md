# Dataset .csv File


The *dataset.csv* file contains all the available information collected from 2 datasets:
- The [Speech Accent Archive](http://accent.gmu.edu/about.php) dataset:
    - 2140 speech samples, utttering the same phrase in English.
    - Speakers from all over the world.
    - Age of the speaker.
    - Country of origin.
    - Age when the speaker started learning English.
    - The birthplace of the speaker.
    - The native language of the speaker.
    - The country in which the speaker resides.
    - The sex of the speaker.
    - There is no explicit information about the accent of the speaker.
- The [openSLR83](http://www.openslr.org/83/) dataset:
    - Around 15000 samples uttering English phrases.
    - Speakers from the UK and Ireland (only male Irish English speakers).
    - The speakers are presumed to be native English speakers.
    - Their age, country of origin and country of residence are unknown.
    - Their sex is known.
    - The speakers have assigned a dialect to their speech themselves:
        - Irish English
        - Midlands English
        - Northern English
        - Southern English
        - Scottish English
        - Welsh English


The information from both datasets is combined into a single .csv file, which contains the following columns:
1. *unnamed*: index column
2. *age*: age of speaker when sample was taken
3. *age_onset*: age when the speaker started learning English
4. *birthplace*: the country of birth of the speaker
5. *filename*: path to the corresponding recording, from the parent folder.
6. *native_language*: the native language of the speaker
7. *sex*: the sex of the speaker
8. *speakerid*: a unique speaker identifier (only for the speech accent archive)
9. *country*: country of residence of the speaker when the sample was taken
10. *eng_speakers*: the percentage of english speakers in the country of residence of the speaker
11. *line*: the utterance in the recording
12. *accent*: the accent label (from openslr + assigned label for the accent archive samples)
13. *accent_group*: same as the accent column, but with some accents grouped together.


The accent of the Speech Accent Archive samples was determined primarily by the native language of the speaker. Native speakers of English were broken down further into the following categories, depending on their birthplace:
- Native English speakers born in the USA were assigned the label *american*
- Born in Canada: *canadian*
- UK: *british*
- Ireland: *irish*
- Australia: *australian*
- Any other country: *english*


The following native speakers were also grouped together in the same accent group:
- Indian accent group (incl. Indo-European Indian languages): Hindi, Bengali, Urdu, Marathi, Punjabi, Gujarati.
- South Slavic: Bulgarian, Croatian, Slovenian.
- West Slavic: Polish, Slovak.
- East Slavic: Russian, Ukrainian.
- Scandinavian: Swedish, Danish, Norwegian.
- Dutch: Dutch, Vlaams, Afrikaans, Frisian.