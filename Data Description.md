# Dataset Description

The competition data comprises transliterations of over 8,000 Old Assyrian cuneiform texts with a comprehensive set of metadata. We provide aligned English translations for a subset of these. We also provide unprocessed texts from almost 900 scholarly publications containing more translations from which you may attempt to create additional training data.

See also the Dataset Instructions for more on the formatting conventions used in these transliterations.

Please note that this is a Code Competition. The data in test.csv is only dummy data to help you author your solutions. When your submission is scored, this example test data will be replaced with the full test set.

# File and Field Information

## train.csv
About 1500 transliterations of Old Assyrian texts from the original excavated tablets each with a translation into English.

- **oare_id**: Identifier in the Old Assyrian Research Environment (OARE) database. Uniquely identifies each text.
- **transliteration**: An Akkadian transliteration of the original tablet text.
- **translation**: A corresponding English translation.

## test.csv
A small example set representative of the test data. When your submission is scored, this example test data will be replaced with the full test set. There are about 4000 sentences in the test data from about 400 unique documents. Note that while the training data has translations aligned at the document level, the test data has translations aligned at the sentence level.

- **id**: A unique identifier for each sentence.
- **text_id**: A unique identifier for each document.
- **line_start, line_end**: Denotes the boundaries of the sentence within the original tablet. Orders the sentences within the document. Note that this field has str type with values like 1, 1', or 1''. See the note on line numbers under Modern Scribal Notations in the Dataset Instructions.
- **transliteration**: An Akkadian transliteration of the original tablet text. Your goal is to produce the corresponding translation.

## sample_submission.csv
A sample submission file in the correct format. See the Evaluation page for more details.

# Supplemental Data

## published_texts.csv
About 8,000 transliterations of Old Assyrian texts together with metadata fields and catalog information from database and museum records as published in the OARE database. You may use these identifiers to retrieve additional information from the linked websites. These transliterations are not provided with translations.

- **oare_id**: Identifier in the OARE database, as in train.csv.
- **online transcript**: URL of the transliteration transcript hosted on the DPI website.
- **cdli_id**: Identifier in the CDLI website(https://cdli.earth/). Multiple IDs are separated by bar |.
- **aliases**: Other published labels for the text (e.g. publication numbers, museum IDs, etc.). Multiple IDs are separated by bar |.
- **label**: Primary designation as a label of the text.
- **publication_catalog**: Labels of the text found in publications and museum records. Multiple IDs are separated by bar |.
- **description**: Basic description of the text.
- **genre_label**: Basic genre assigned to the text. Not available for all texts.
- **inventory_position**: Label of the text as found in the museum. Multiple IDs are separated by bar |.
- **online_catalog**: URL of the Yale collection with CC-0 metadata and images.
- **note**: Notes made by specialists for commentary or translations.
- **interlinear_commentary**: References to publications which discuss the text at specific lines.
- **online_information**: URL of the text in the British Museum (note these images are copyright of the British Museum, not in CC). Not available for all texts.
- **excavation_no**: Identifier assigned to the text from the excavation.
- **oatp_key**: Identifier assigned by the Old Assyrian Text Project.
- **eBL_id**: Identifier in the eBL website(https://www.ebl.lmu.de/library/).
- **AICC_translation**: URL of the first published online(https://aicuneiform.com/search?q=P361099) machine translation. Note that most of these translations are very poor quality.
- **transliteration_orig**: Original text transliteration from the OARE database.
- **transliteration**: Clean version of the text transliteration based on these formatting suggestions.

## publications.csv
Contains the raw text of about 880 scholarly publications containing translations from Old Assyrian into multiple modern languages. The texts were produced via OCR with LLM postprocessing. You may attempt to extract these translations and align them with the transliterations in published_texts.csv. Note that the translations often are given in a language other than English.

- **pdf_name**: The name of the PDF file from which the text was extracted.
- **page**: The page number where the given text occured.
- **page_text**: The text of the article itself.
- **has_akkadian**: Whether or not the text contains Akkadian transliterations.

## bibliography.csv
Bibliographic data for the texts in publications.csv.

- **pdf_name**: An ID corresponding to that in publications.csv.
- **title, author, author_place, journal, volume, year, pages**: Standard bibliographic data.

## OA_Lexicon_eBL.csv
This file contains a list of all the Old Assyrian words in transliteration with their lexical equivalents (that is, how they are found in a dictionary). The links included are to an online Akkadian dictionary hosted by LMU, the electronic Babylonian Library(https://www.ebl.lmu.de/dictionary) (eBL).

- **type**: Type of word (e.g. word, PN = person name, GN = geographic name).
- **form**: String-literal word, as found in transliteration.
- **norm**: Normalized form, with hyphens removed and vowel length indications.
- **lexeme**: Lemmatized form, as found in a dictionary.
- **eBL**: URL of the online dictionaries in the electronic Babylonian Library (eBL).
- **I_IV**: Roman numeral designation of the homonym lexemes, corresponding to the Concise Dictionary of Akkadian (CDA) found in the eBL.
- **A_D**: Alphabetic designation of the homonym lexemes, corresponding to the Chicago Assyrian Dictionary(https://isac.uchicago.edu/research/publications/chicago-assyrian-dictionary).
- **Female(f)**: Designation for female gender.
- **Alt_lex**: Alternative normalizations.

## eBL_Dictionary.csv
The complete dictionary of Akkadian words from the eBL database. It collects the data provided by the URLs at eBL in the OA_Lexicon_eBL.csv file.

## resources.csv
A list of resources that might be used for additional data.

## Sentences_Oare_FirstWord_LinNum.csv
An aid to aligning translations at the sentence level for the data in train.csv. Indicates the first word of each sentence and its location on the tablet.

# Suggested Workflow for Building Additional Training Data

The publications.csv file contains OCR output from almost 900 PDFs, and extracting the translations from these texts is an essential first step. Before any machine learning can happen, the training data needs to be reconstructed and aligned. Here’s a simple path you can follow:

1. **Locate each text and its translation**: Use the document identifiers (IDs, aliases, or museum numbers) to match transliterations with their corresponding translations in the OCR output.
2. **Convert all translations to English**: The source translations may appear in multiple languages (e.g., English, French, German, Turkish). For consistency, convert everything to English.
3. **Create sentence-level alignments**: Break both the Akkadian transliteration and the matching English translation into sentences and align them pairwise. This sentence-level mapping is the most useful format for training and evaluating machine translation models.

Once these steps are completed, you’ll have a clean, aligned dataset ready for machine learning.

# Bibliography

The bibliography reflects the secondary sources we used to retrieve the translations for the challenge. Because they are held in different copyrights, we suggest each work should be cited if they were used when generating machine translations.

Additional bibliography citations for the primary sources can be found here:

- https://cdli.earth/publications
- https://cdli.ox.ac.uk/wiki/abbreviations_for_assyriology