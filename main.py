import argparse
from io import TextIOWrapper
import logging
import string
import subprocess
from typing import List, Optional
import unicodedata
import os
import nltk
from nltk import pos_tag, word_tokenize

# Download required NLTK resources if not already available
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('punkt', quiet=True)

def is_verb_in_sentence(word, sentence):
    # Tokenize the sentence into words
    tokens = word_tokenize(sentence)

    # Get POS tags for each word in the sentence
    tagged_words = pos_tag(tokens)

    # Find the POS tag of the target word in the tagged words list
    for tagged_word, pos in tagged_words:
        if tagged_word.lower() == word.lower():  # Case insensitive match
            return pos.startswith('VB')  # Check if the tag starts with 'VB' (verb)

    return False  # Word not found in sentence


ipa_vowels = "aeiouɑɒæɛɪʊʌɔœøɐɘəɤɨɵɜɞɯɲɳɴɶʉʊʏ"
ipa_consonants = "pbtdkgqɢʔmɱnɳɲŋɴʙrʀⱱɾɽɸβfvθðszʃʒʂʐçʝxɣχʁħʕhɦɬɮʋɹɻjɰlɭʎʟɝ"
ipa_letters = ipa_vowels+ipa_consonants


normal_reductions = {"for": "fɝ", "your": "jɝ", "you're": "jɝ", "and": "ən", "an": "ən", "that": "ðət",
 "you": "jə", "do": "də", "at": "ət", "from": "fɹəm",
 "there": "ðɝ", "they're": "ðɝ", "when": "wən", "can": "kən", "into": "ɪndə", "some": "səm",
 "than": "ðən", "then": "ðən", "our": "ɝ", "because": "kəz", "us": "əs", "such": "sətʃ", "as": "əz", "i'll": "əl",
 "you'll": "jəl", "he'll": "hɪl", "she'll": "ʃɪl", "it'll": "ɪtəl", "we'll": "wɪl", "they'll": "ðəl"}

#reduction that only happen if prev word ends with a consonant
h_reduction = {"him": "ɪm", "his": "ɪz", "her": "ɝ", "he":"i", "who": "u", "have": "əv", "has": "əz", "them": "əm"}
# I think these next reduction are often not reduced, so not adding those:
# "but": "bət", "one": "ən", "so": "sə",

double_word_reductions = { "do you": "dju", "what did": "wʌd", "he has": "hiz", "she has": "ʃiz", "it has": "ɪts",
 "i will": "əl", "you will": "jəl", "he will": "hɪl", "she will": "ʃɪl", "it will": "ɪtəl",
 "we will": "wɪl", "they will": "ðəl", "you have": "juv", "we have": "wɪv", "they have": "ðeɪv",
 "i am": "aɪm", "he is": "hiz", "she is": "ʃiz", "it is": "ɪts", "we are": "wɝ", "want to": "wɑnə",
 "kind of": "kaɪndə", "give me": "ɡɪmi", "let me": "lemi" }
# These next few are not reduced often (For example - I should have it). I'll only reduce those before a verb, though I'm not sure it's the right call
double_word_with_verb = {"could have": "cʊdə", "should have": "ʃʊdə", "would have": "wʊdə", "going to": "gɑnə"}
# all noun+will can be reduced to x'll, but too hard for me to implement

# Most of those are not wrong, we just prefer it like this. These will be replced if appear in a word (good for plural and such):
improved_pronounciations = {
    "fæməli":"fæmli",
    "kʌmfɝtəbəl":"kʌmftɝbəl",
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
    "kwɔɹtɝ":"kɔɹtɝ",
    "pɹɪzənɝ":"pɹɪznɝ",
    # flight mistakes:
    "heɪˈvɛnt":"hævənt",
    "hæsnt":"hæzənt",
    "ʌnˈmindfəl": "ʌnˈmaɪndfəl",
    "junɪˈdɛntəˈfaɪəbəl": "ʌnaɪdentəˈfaɪəbəl",
    # words specifically for certain books:
    "waɪtɪˈkloʊks": "waɪtˈkloʊks",
    "waɪtɪˈkloʊk": "waɪtˈkloʊk",
}

def get_next_char(text: list, word_idx: int, letter_idx: int) -> str:
    word = text[word_idx]
    if len(word) > letter_idx + 1:
        return word[letter_idx + 1]
    if len(text) > word_idx + 1:
        if text[word_idx + 1] == "":
            return ""
        return text[word_idx + 1][0]
    return ""

def get_prev_char(text: list, word_idx: int, letter_idx: int) -> str:
    word = text[word_idx]
    if letter_idx > 0:
        return word[letter_idx - 1]
    if word_idx > 0:
        if text[word_idx - 1] == "":
            return ""
        return text[word_idx - 1][-1]
    return ""


def add_reductions_with_stress(ipa_text: str, original_text: str):
    # here we still have the stress sine, and t/d's weren't handled yet
    out_text = ipa_text.split(" ")
    original_arr = original_text.lower().split(" ")
    for i, word in enumerate(out_text):
        stripped = word.strip()
        next_char = get_next_char(out_text, i, len(word)-1)
        prev_char = get_prev_char(out_text, i, 0)
        if stripped == "ʌv":
            # check if both next word starts with a consonant and prev word ends with one
            if prev_char != "" and prev_char != "" and prev_char in ipa_consonants and next_char in ipa_consonants:
                out_text[i] = word.replace("ʌv",  "ə")
            continue
        elif original_arr[i] in normal_reductions.keys() or original_arr[i] in h_reduction.keys():
            # validate that this is not the last word. last word in sentence don't get reduced
            if next_char != "":
                if original_arr[i] in normal_reductions.keys():
                    out_text[i] = normal_reductions[original_arr[i]]
                elif prev_char != "" and prev_char in ipa_consonants:
                    # h reductions happend only after a consonant
                    out_text[i] = h_reduction[original_arr[i]]
        for orig, changed in improved_pronounciations.items():
            if orig in stripped:
                out_text[i] = word.replace(orig, changed)
                break
    return " ".join(out_text)

def add_double_word_reductions(ipa_text: str, original_text: str):
    out_arr = ipa_text.split(" ")
    original_arr = original_text.lower().split(" ")
    removed_words = 0
    for i in range(len(original_arr)):
        original_word = original_arr[i]
        for orig, changed in list(double_word_reductions.items())+list(double_word_with_verb.items()):
            first = orig.split(" ")[0]
            if original_word == first:
                second = orig.split(" ")[1]
                if len(original_arr) > i and original_arr[i+1] == second:
                    # validate that this is not the last word. last word in sentence don't get reduced
                    next_char = get_next_char(original_arr, i+1, len(second)-1)
                    if next_char != "":
                        if orig in double_word_with_verb:
                            if not is_verb_in_sentence(original_arr[i+2], original_text):
                                continue

                        out_arr[i - removed_words] = changed
                        del out_arr[i - removed_words + 1]
                        removed_words += 1
    return " ".join(out_arr)

def handle_t_d(ipa_text: str):
    # True t/d - beggining of a word or a stressed syllable
    # Dropped t/d - after n, unless syllable split between the n/r (until, intense).
                # between to consanants. not after r (partly)
    # Flap t/d - between 2 vowels (r counts on the left of the t) unless the t starts a stressed syllable

    # t/d becomes ɾ (tap or flap d) when between 2 vowels (r also counts) (linking words if needed) "I told you?" not at first letter of word

    # t/d between two consonants (r doesn't count) can be removed. t/d after n as well sometimes? identify, twenty, want, count, disappoint.
    #   not in into, entry, antique, intend, contain, intake, intonation (first syllable is unstressed in most these words. All except entry)
    # tr -> tʃɹ, dr - dʒɹ, tj -> tʃj, dj - dʒj
    out_text = ipa_text.replace("tɹ", "tʃɹ").replace("dɹ", "dʒɹ").replace("tj", "tʃj").replace("dj", "dʒj").split(" ")
    for i, word in enumerate(out_text):
        if word in ("ɹænd", "ɹændz", "mæt"):
            continue #names that annoyingly gets reduced and we want to skip
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
            if prev_letter == 'n' and next_letter != 'ʃ' and next_letter != 'ʒ' and prev_letter != next_letter:
                if letter_idx != len(out_word) - 1 and next_letter in ipa_vowels + "ɝ":
                    # in the middle of a word, we don't seem to drop t/d before vowels
                    continue
                # drop t/d
                out_word = out_word[:letter_idx] + out_word[letter_idx+1:]
                break
            if prev_letter in (ipa_vowels + "ɝɹ") and next_letter in (ipa_vowels + "ɝ"):
                # between 2 vowels, flap t/d
                out_word =  out_word[:letter_idx] + 'ɾ' + out_word[letter_idx+1:]
                break
            if prev_letter in (ipa_consonants) and prev_letter not in "ɝɹʃ":
                if next_letter in (ipa_consonants) and next_letter not in "ɝɹʃ":
                    if prev_letter != next_letter:
                        # between two consonants, drop t/d
                        out_word = out_word[:letter_idx] + out_word[letter_idx+1:]
                        break

        out_text[i] = out_word
    return " ".join(out_text)


def normalize(text: str):
    text = text.replace("’", "'").replace("‘", "'").replace('”', '"').replace('“', '"').replace("—", " - ")
    text = unicodedata.normalize('NFD', text)
    text = ''.join(filter(lambda x: x in string.printable, text))
    return text

# Words that are wrong in the original text
nn_words = {'fonnula', 'fanngirls', 'annorers', 'fishennans', 'speannen', 'fonning', 'bannaids', 'outennost', 'unanned', 'anns', 'alanned', 'perfonning', 'tenn',
  'fonner', 'fannyard', 'intenneshed', 'indetenninate', 'munnurs', 'fannhouses', 'poleanns', 'finned', 'redann', 'fannboy',  'poleann', 'intenninably',
   'perfonnance', 'transfonns', 'rainstonns', 'pennanently', 'rainstonn', 'channing', 'fonned', 'hannony', 'fanner', 'annload', 'ennine', 'infonn',
   'annariz', 'pennanent', 'chann', 'swann', 'stonns', 'wanning', 'stronganns', 'swanns', 'annloads', 'fonn', 'perfonners', 'detennined', 'fonnidable',
   'hanned', 'fonnations', 'unifonned', 'uppennost',  'wonnwood', 'bordennan', 'infonnative', 'annchairs', 'annbands', 'infonning', 'andonnan', 'fonns',
   'transfonned', 'ovennatched', 'countennanding', 'innkeepera',  'swanned', 'penneated', 'nonnal', 'infonnal', 'wannest', 'bordennen', 'gannents', 'wannly',
   'anned', 'tonnentor', 'fonnalities', 'hailstonn', 'annor', 'redanns', 'stonn', 'annour', 'insunnountable', 'fannwife', 'gannent','perfonner', 'rivennan',
   'hanns', 'alann', 'alanningly', 'fannhouse', 'reannes', 'undennined', 'tonn', 'annsmens', 'tyrannizing', 'munnured', 'perfonned', 'hann', 'infonnants',
   'wonned', 'tunnoil', 'detennine', 'detennina', 'fonnal',  'windstonn', 'munnuring', 'detenninedly', 'wann', 'annband', 'anny', 'confinned', 'finnly',
   'infonnation', 'ann', 'fanning', 'intennittently','annored', 'annsmen', 'annsman', 'hanning', 'fonnality', 'hannless', 'annpit', 'sideanned', 'pennit',
   'runnerhan', 'skinnish', 'sheepfanner', 'fanns', 'wannth', 'annies', 'annys', 'perfonnedat', 'aftennath','tonnent', 'thinning', 'tenns', 'unhanned',
   'skinnishes', 'swanning', 'kennit','fonnidably', 'thunderstonn', 'perfonn', 'unifonn', 'finn', 'fanners', 'pennitted', 'fonnerly', 'fann',  'vennin',
   'foreann', 'detennination', 'annchair', 'jenn', 'wanned', 'wanner', 'nonnality', 'platfonn', 'wonn', 'wonns', 'andonnen'}

m_words = {"comer", "comers", "bum", "bums", "bam", "bom", "hom", "leam", "stem","wom"}
def fix_nn(text: str):
    out = text.split(" ")
    for i, word in enumerate(out):
        stripped_word = ''.join([char for char in word if char.isalpha()])
        if stripped_word.lower() in nn_words:
            out[i] = word.replace("nn", "rm")
        elif stripped_word.lower() in m_words:
            out[i] = word.replace("m", "rn")
    return out

def fix_numbers(text_arr: List[str]):
    i = 0
    while i < len(text_arr):
        word = text_arr[i]
        if i > 0 and text_arr[i-1] == "chapter" or "(" in word or ")" in word:
            i+=1
            continue
        if word == "1":
            text_arr[i] = "I"
        elif word == '"1':
            text_arr[i] = '"I'
        elif word in ("7", "7/", '"7'):
            if i == 0:
                print(f"problem!!!! {text_arr}")
            else:
                text_arr[i-1] = text_arr[i-1] + "'t"
                del text_arr[i]
                continue # don't increase i
        elif word in ("7....", "7."):
            text_arr[i-1] = text_arr[i-1] + "'" + word.replace("7", "t")
            del text_arr[i]
            continue # don't increase i
        elif "11" in word:
            text_arr[i] == word.replace("11", "ll")
        i+=1
    return text_arr



def run_flite(text: str):
    fixed_text = text
    # fixed_text = " ".join(fix_numbers(fix_nn(text.lower())))
    try:
        ipa_text = subprocess.check_output(['flite', "-t", fixed_text, "-i"]).decode('utf-8')
    except OSError:
        logging.warning('lex_lookup (from flite) is not installed.')
        ipa_text = ''
    except subprocess.CalledProcessError:
        logging.warning('Non-zero exit status from lex_lookup.')
        ipa_text = ''

    ipa_text = add_reductions_with_stress(ipa_text, fixed_text)
    ipa_text = add_double_word_reductions(ipa_text, fixed_text)
    #from here on out, fixed_text can no longer be trusted (length doesn't match the ipa_text length)
    ipa_text = handle_t_d(ipa_text)
    #remove stress marks
    ipa_text = ipa_text.replace("ˈ", "")
    return fixed_text, ipa_text

sentence_enders = '''.!?'")]}:;>0123456789'''

cached_text = ""
line_end_count = 0
is_chapter = False
def fix_line_ending(line: str) -> Optional[str]:
    """returns None if we should skip flite and go to the next word. Otherwise returns the text to parse"""
    global is_chapter
    global cached_text
    global line_end_count
    stripped = line.strip()
    if len(stripped) > 0 and stripped[-1] not in sentence_enders:
        if stripped in ("PROLOGUE", "CHAPTER", "EPILOGUE"):
            is_chapter = True
            temp = cached_text
            cached_text = "\n" + line[:-1] + " "
            return None if temp == "" else temp

        cached_text += line.replace("\n", "")
        if is_chapter:
            is_chapter = False
            temp = cached_text + "\n\n"
            cached_text = ""
            return temp

        cached_text += " "
        return None
    # If we reached here, either line is "\n" or it ends with an endmark. is_missing_endmark refers to the prev line now
    if cached_text != "":
        if line == "\n":
            return None
        line = cached_text + line
        cached_text = ""
        line_end_count = 0
        return line
    if line == "\n":
        line_end_count += 1
        if line_end_count > 1:
            return None
        return line
    line_end_count = 0
    is_chapter = False
    return line

def print_ipa(out_file: Optional[TextIOWrapper], lines: List[str], fix_line_ends: bool = True):
    global cached_text
    for line in lines:
        normalized_line = normalize(line)
        if fix_line_ends:
            normalized_line = fix_line_ending(normalized_line)
            if normalized_line is None:
                continue
        if out_file:
            if normalized_line == "\n":
                out_file.write(normalized_line)
                continue
            orig, ipa = run_flite(normalized_line)
            out_file.write(ipa)
            out_file.write(orig)
        else:
            print(run_flite(normalized_line))
    if cached_text != "":
        if out_file:
            orig, ipa = run_flite(cached_text)
            out_file.write(ipa)
            out_file.write(orig)
        else:
            print(run_flite(cached_text))




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

