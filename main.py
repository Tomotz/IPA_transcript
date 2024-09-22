import argparse
from io import TextIOWrapper
import logging
import string
import subprocess
from typing import Optional
import unicodedata
import os

import spacy

# Load the English model
nlp = spacy.load('en_core_web_sm')


ipa_vowels = "aeiouɑɒæɛɪʊʌɔœøɐɘəɤɨɵɜɞɯɲɳɴɶʉʊʏ"
ipa_consonants = "pbtdkgqɢʔmɱnɳɲŋɴʙrʀⱱɾɽɸβfvθðszʃʒʂʐçʝxɣχʁħʕhɦɬɮʋɹɻjɰlɭʎʟɝ"
ipa_letters = ipa_vowels+ipa_consonants

# last word in sentence don't get reduced
# for -> fəɹ
# to -> tə (already done by flite)
# You're -> jəɹ
# Your -> jəɹ
# They're -> ðəɹ
# and ->ən
# an ->ən
# that -> ðət
# he has - hiz
# She has -> ʃiz
# it has -> ɪts
# He - i (linknig to the previous word. Only after consonant)
# As - əz
# You - jə
# Do - də
# at -> ət
# but - bət
# his -> ɪz
# her -> əɹ
# from -> fɹəm
# or -> əɹ
# I will -> aɪl -> əl
# you will -> jul -> jəl
# he will -> hil-> hɪl
# she will -> ʃil-> ʃɪl
# it will -> ɪɾəl
# we will -> wil-> wɪl
# they will -> ðəl
# all noun+will can be reduced to x'll
# one -> ən
# would - wəd
# should - ʃəd
# could - kəd
# There/They're (ðɛəɹ) - ðəɹ
# What - if the next word starts with a d, we can reduce (linkning) - wə d
# So - sə
# Who - u
# When - wən
# Can (kæn) - kən
# him -> əm
# into -> ɪndə
# some -> səm
# them - əm
# than - ðən
# then - ðən
# our - əɹ
# because - bɪkəz - kəz
# us - əs
# such - sətʃ


# want to - wanna, going to - gonna, kind of - kinda, could/should/would have - coulda/shoulda/woulda, give me - gimme, let me - lemme
# you have -juv
# We have - wɪv
# They have - ðeɪv
# I am -> aɪm
# He is -> hiz
# She is -> ʃiz
# it is -> ɪts
# We are ->wəɹ



# Those are not wrong, we just prefer it like this. These will be replced if appear in a word (good for plural and such):
improved_pronounciations = {
    "fæməli":"fæmli",
    "kʌmfɝtəbəl":"kʌmftəɹbəl",
    "feɪvɝɪt":"feɪvɹɪt",
    "pɹɑbəbli":"pɹɑbli",
    "dɪfɝənt":"dɪfɹənt",
    "kæmɝə":"kæmɹə",
    "lɪsənɪŋ":"lɪsnɪŋ",
    "mɛmɝi":"mɛmɹi",
    "tɹævəlɪŋ":"tɹævlɪŋ",
    "nætʃɝəl":"nætʃɹəl",
    "æktʃəwəli":"æktʃəli ",
    "ɹɛstɝˈɑnt":"ɹɛstˈɹɑnt",
    "kwɔɹtɝ":"kɔɹtɝ"
}

def get_next_char(text, word_idx, letter_idx):
    word = text[word_idx]
    if len(word) > letter_idx + 1:
        return word[letter_idx + 1]
    if len(text) > word_idx + 1:
        return text[word_idx + 1][0]
    return ""
    
def add_reductions_with_stress(text: str):
    # here we still have the stress sine, and t/d's weren't handled yet
    out_text = text.split(" ")
    for i, word in enumerate(out_text):
        stripped = word.strip()
        if stripped == "ʌv":
            # check if both next and prev words starts with a consonant
            if i > 0 and len(out_text) > i + 1 and out_text[i-1][-1] in ipa_consonants and out_text[i+1][0] in ipa_consonants:
                out_text[i] = word.replace("ʌv",  "ə")
            continue
        for orig, changed in improved_pronounciations.items():
            if orig in stripped:
                out_text[i] = word.replace(orig, changed)
                break
    return " ".join(out_text)

def handle_flap_t_d(text: str):
    # True t/d - beggining of a word or a stressed syllable
    # Dropped t/d - after n, unless syllable split between the n/r (until, intense).
                # between to consanants. not after r (partly)
    # Flap t/d - between 2 vowels (r counts on the left of the t) unless the t starts a stressed syllable

    # t/d becomes ɾ (tap or flap d) when between 2 vowels (r also counts) (linking words if needed) "I told you?" not at first letter of word
        
    # t/d between two consonants (r doesn't count) can be removed. t/d after n as well sometimes? identify, twenty, want, count, disappoint. 
    #   not in into, entry, antique, intend, contain, intake, intonation (first syllable is unstressed in most these words. All except entry)
    # tr -> tʃɹ, dr - dʒɹ
    out_text = text.replace("tɹ", "tʃɹ").replace("dɹ", "dʒɹ").split(" ")
    for i, word in enumerate(out_text):
        out_word = word
        for letter_idx in range(1, len(out_word)): # Beggining of a word will have a true t/d, so start from 1
            letter = out_word[letter_idx]
            if letter != 't' and letter != 'd':
                continue
            prev_letter = out_word[letter_idx - 1]
            if prev_letter == 'ˈ':
                # stressed syllable. True t/d
                continue
            next_letter = get_next_char(out_text, i, letter_idx)
            if next_letter == "" or next_letter not in ipa_letters:
                # end of sentence. Prefer True t/d
                continue
            # note that if we changed the word so we can't keep iterrating it. We will lose any other t/d changes, but that's rare so so be it.
            if prev_letter == 'n' and next_letter != 'ʃ': 
                # drop t/d
                out_word = out_word[:letter_idx] + out_word[letter_idx+1:]
                break 
            if prev_letter in (ipa_vowels + "ɝɹ") and next_letter in (ipa_vowels + "ɝ"):
                # between 2 vowels, flap t/d
                out_word =  out_word[:letter_idx] + 'ɾ' + out_word[letter_idx+1:] 
                break
            if prev_letter in (ipa_consonants) and prev_letter not in "ɝɹʃ":
                if next_letter in (ipa_consonants) and next_letter not in "ɝɹʃ":
                    # between two consonants, drop t/d
                    out_word = out_word[:letter_idx] + out_word[letter_idx+1:]
                    break

        out_text[i] = out_word
    return " ".join(out_text)


def spacy_tokenize(sentence: str):
    
    # print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
    #             token.shape_, token.is_alpha, token.is_stop)
    # Text: The original word text.
    # Lemma: The base form of the word.
    # POS: The simple UPOS part-of-speech tag.
    # Tag: The detailed part-of-speech tag.
    # Dep: Syntactic dependency, i.e. the relation between tokens.
    # Shape: The word shape – capitalization, punctuation, digits.
    # is alpha: Is the token an alpha character?
    # is stop: Is the token part of a stop list, i.e. the most common words of the language?


    doc = nlp(sentence)

    # Print grammatical components
    for token in doc:
        print(f"Word: {token.text}, POS: {token.pos_}, Dep: {token.dep_}")

    for token in doc:
        if token.pos_ in ["AUX", "PRON", "ADP", "DET"]:
            print(f"Word '{token.text}' is likely reduced.")




def normalize(text: str):
    text = unicodedata.normalize('NFD', text)
    text = ''.join(filter(lambda x: x in string.printable, text))
    return text

def run_flite(text: str):
    spacy_tokenize(text)
    normalized_text = normalize(text).lower()
    try:
        ipa_text = subprocess.check_output(['flite', "-t", normalized_text, "-i"]).decode('utf-8')
    except OSError:
        logging.warning('lex_lookup (from flite) is not installed.')
        ipa_text = ''
    except subprocess.CalledProcessError:
        logging.warning('Non-zero exit status from lex_lookup.')
        ipa_text = ''
    
    ipa_text = add_reductions_with_stress(ipa_text)
    ipa_text = handle_flap_t_d(ipa_text)
    #remove stress marks
    ipa_text = ipa_text.replace("ˈ", "")
    return ipa_text

def print_ipa(out_file: Optional[TextIOWrapper], lines: str):
    for line in lines:
        if out_file:
            out_file.write(run_flite(line))
            out_file.write(line)
        else:
            print(run_flite(line))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str, help="Input text or filename")
    parser.add_argument("-f", "--file", action="store_true", 
                        help="Indicate that the input is a filename/dirname instead of text. If dir, will translate all the files in that dir. In this case, output must be given, and be a directory")
    parser.add_argument("-o", "--output", type=str, nargs='?', default=None, help="Optional output file/directory. If not given, will print to stdout")
    
    # Parse the arguments
    args = parser.parse_args()
    out_file = None
    if args.file:
        if os.path.isfile(args.data):
            lines = open(args.data).readlines()
        else:
            assert args.output, "When directory is given, output must also be a directory"
            for root, folders, files in os.walk(args.data):
                for file_name in files:
                    with open(os.path.join(root, file_name)) as f:
                        lines = f.readlines()
                    out_file_name = "ipa_" + file_name
                    with open(os.path.join(args.output, out_file_name), "w") as o:
                        print_ipa(o, lines)
            return
    else:
        lines = args.data.split("\n")
    if args.output is not None:
        out_file = open(args.output, "w")
    print_ipa(out_file, lines)
    if args.output is not None:
        out_file.close()

if __name__ == "__main__":
    main()

    