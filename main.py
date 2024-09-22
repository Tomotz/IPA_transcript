import argparse
from io import TextIOWrapper
import logging
import string
import subprocess
from typing import Optional
import unicodedata
import os


ipa_vowels = "aeiouɑɒæɛɪʊʌɔœøɐɘəɤɨɵɜɞɯɲɳɴɶʉʊʏ"
ipa_consonants = "pbtdkgqɢʔmɱnɳɲŋɴʙrʀⱱɾɽɸβfvθðszʃʒʂʐçʝxɣχʁħʕhɦɬɮʋɹɻjɰlɭʎʟɝ"
ipa_letters = ipa_vowels+ipa_consonants

# last word in sentence don't get reduced
normal_reductions = {"for": "fəɹ", "your": "jəɹ", "you're": "jəɹ", "and": "ən", "an": "ən", "that": "ðət", 
 "you": "jə", "do": "də", "at": "ət", "from": "fɹəm", "or": "əɹ", "would": "wəd", "should": "ʃəd", "could": "kəd", 
 "there": "ðəɹ", "they're": "ðəɹ", "when": "wən", "can": "kən", "into": "ɪndə", "some": "səm", 
 "them": "əm",  "than": "ðən", "then": "ðən", "our": "əɹ", "because": "kəz", "us": "əs", "such": "sətʃ", "i'll": "əl",
 "you'll": "jəl", "he'll": "hɪl", "she'll": "ʃɪl", "it'll": "ɪtəl", "we'll": "wɪl", "they'll": "ðəl"}
h_reduction = {"him": "ɪm", "his": "ɪz", "her": "əɹ", "he":"i", "who": "u", "have": "əv", "has": "əz"}
# I think these next reduction are often not reduced, so not adding those:
# "as": "əz", "but": "bət", "one": "ən", "so": "sə", 

double_word_reductions = { "do you": "dju", "what did": "wʌd", "he has": "hiz", "she has": "ʃiz", "it has": "ɪts",
 "i will": "əl", "you will": "jəl", "he will": "hɪl", "she will": "ʃɪl", "it will": "ɪtəl", 
 "we will": "wɪl", "they will": "ðəl", "you have": "juv", "we have": "wɪv", "they have": "ðeɪv", 
 "i am": "aɪm", "he is": "hiz", "she is": "ʃiz", "it is": "ɪts", "we are": "wəɹ", "want to": "wɑnə",
 "going to": "gɑnə", "kind of": "kaɪndə", "could have": "cʊdə", "sould have": "ʃʊdə", "would have": "wʊdə",
 "give me": "ɡɪmi", "let me": "lemi" }
# all noun+will can be reduced to x'll, but too hard for me to implement

# Most of those are not wrong, we just prefer it like this. These will be replced if appear in a word (good for plural and such):
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
    "kwɔɹtɝ":"kɔɹtɝ",
    # Flight mistakes:
    "heɪˈvɛnt":"hævənt",
    "hæsnt":"hæzənt",
    "ʌnˈmindfəl": "ʌnˈmaɪndfəl",
    "junɪˈdɛntəˈfaɪəbəl": "ʌnaɪdentəˈfaɪəbəl",
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
        if word in 
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
            if prev_letter == 'n' and next_letter != 'ʃ' and prev_letter != next_letter: 
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

def fix_numbers(text_arr: str):
    i = 0
    while i < len(text_arr):
        word = text_arr[i]
        if i > 0 and text_arr[i-1] == "Chapter" or "(" in word or ")" in word:
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
    normalized_text = normalize(text)
    fixed_text = normalized_text
    # fixed_text = " ".join(fix_numbers(fix_nn(normalized_text.lower())))
    try:
        ipa_text = subprocess.check_output(['flite', "-t", fixed_text, "-i"]).decode('utf-8')
    except OSError:
        logging.warning('lex_lookup (from flite) is not installed.')
        ipa_text = ''
    except subprocess.CalledProcessError:
        logging.warning('Non-zero exit status from lex_lookup.')
        ipa_text = ''
    
    ipa_text = add_reductions_with_stress(ipa_text, fixed_text)
    ipa_text = handle_t_d(ipa_text)
    #remove stress marks
    ipa_text = ipa_text.replace("ˈ", "")
    return fixed_text, ipa_text
   


def print_ipa(out_file: Optional[TextIOWrapper], lines: str):
    for line in lines:
        if out_file:
            if line == "\n":
                out_file.write(line)
                continue
            orig, ipa = run_flite(line)
            out_file.write(ipa)
            out_file.write(orig)
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

    