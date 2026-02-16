# Rebuild flite: cd flite; make clean && make -j$(nproc)

import argparse
from concurrent.futures import ThreadPoolExecutor
from io import TextIOWrapper
import html as html_module
import json
import logging
import re
import string
import subprocess
import sys
from typing import Dict, List, Optional, Tuple
import unicodedata
import os
import nltk
from nltk import pos_tag, word_tokenize

# Download required NLTK resources if not already available
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

def is_verb_in_sentence(word, sentence):
    tokens = word_tokenize(sentence)
    tagged_words = pos_tag(tokens)
    word_lower = word.lower()
    for tagged_word, pos in tagged_words:
        if tagged_word.lower() == word_lower:
            return pos.startswith('VB')
    return False


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

double_word_reductions = { "do you": "dju", "what did": "wʌd", # These are ofter wrong: "he has": "hiz", "she has": "ʃiz", "it has": "ɪts",
 "i will": "əl", "you will": "jəl", "he will": "hɪl", "she will": "ʃɪl", "it will": "ɪtəl",
 "we will": "wɪl", "they will": "ðəl", # These are ofter wrong: "you have": "juv", "we have": "wɪv", "they have": "ðeɪv",
 "i am": "aɪm", "he is": "hiz", "she is": "ʃiz", "it is": "ɪts", "we are": "wɝ", "want to": "wɑnə",
 "kind of": "kaɪndə", "give me": "ɡɪmi", "let me": "lemi" }
# These next few are not reduced often (For example - I should have it). I'll only reduce those before a verb, though I'm not sure it's the right call
double_word_with_verb = {"could have": "cʊdə", "should have": "ʃʊdə", "would have": "wʊdə", "going to": "gɑnə"}
# all noun+will can be reduced to x'll, but too hard for me to implement

_double_word_lookup: Dict[str, List[Tuple[str, str, bool]]] = {}
for _orig, _changed in double_word_reductions.items():
    _first, _second = _orig.split(" ")
    _double_word_lookup.setdefault(_first, []).append((_second, _changed, False))
for _orig, _changed in double_word_with_verb.items():
    _first, _second = _orig.split(" ")
    _double_word_lookup.setdefault(_first, []).append((_second, _changed, True))

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
    "æktʃəwəli":"æktʃəli",
    "ɹɛstɝˈɑnt":"ɹɛstˈɹɑnt",
    "ɛvɝi":"ɛvɹi",
    "dʒɛnɝəl":"dʒɛnɹəl",
    "ævɝɪdʒ":"ævɹɪdʒ",
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
        if original_word not in _double_word_lookup:
            continue
        for second, changed, needs_verb in _double_word_lookup[original_word]:
            if len(original_arr) > i + 1 and original_arr[i+1] == second:
                next_char = get_next_char(original_arr, i+1, len(second)-1)
                if next_char != "":
                    if second in ("will", "have", "has") and i + 2 < len(original_arr) and original_arr[i+2] == "not":
                        continue
                    if needs_verb:
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

_flite_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'flite', 'bin', 'flite')

def _call_flite(text: str) -> str:
    try:
        return subprocess.check_output([_flite_path, "-t", text, "-i"]).decode('utf-8')
    except OSError:
        logging.warning('lex_lookup (from flite) is not installed.')
        return ''
    except subprocess.CalledProcessError:
        logging.warning('Non-zero exit status from lex_lookup.')
        return ''

def run_flite(text: str):
    fixed_text = text
    # fixed_text = " ".join(fix_numbers(fix_nn(text.lower())))
    ipa_text = _call_flite(fixed_text)

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

CHECKPOINT_INTERVAL = 10

def get_checkpoint_path(output_path):
    if os.path.isdir(output_path):
        return os.path.join(output_path, ".ipa_checkpoint")
    return output_path + ".ipa_checkpoint"

def load_checkpoint(checkpoint_path):
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            return json.load(f)
    return {}

def save_checkpoint(checkpoint_path, data):
    with open(checkpoint_path, "w") as f:
        json.dump(data, f)

def remove_checkpoint(checkpoint_path):
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

FLITE_BATCH_SIZE = 32
FLITE_MAX_WORKERS = 8

def _run_flite_batch(texts: List[str]) -> List[Tuple[str, str]]:
    with ThreadPoolExecutor(max_workers=FLITE_MAX_WORKERS) as executor:
        ipa_results = list(executor.map(_call_flite, texts))
    results = []
    for fixed_text, ipa_text in zip(texts, ipa_results):
        ipa_text = add_reductions_with_stress(ipa_text, fixed_text)
        ipa_text = add_double_word_reductions(ipa_text, fixed_text)
        ipa_text = handle_t_d(ipa_text)
        ipa_text = ipa_text.replace("ˈ", "")
        results.append((fixed_text, ipa_text))
    return results

def print_ipa(out_file: Optional[TextIOWrapper], lines: List[str], fix_line_ends: bool = True, checkpoint_path: Optional[str] = None, start_line: int = 0):
    global cached_text
    total = len(lines)

    pending_texts: List[str] = []
    pending_indices: List[int] = []
    newline_positions: List[Tuple[int, str]] = []

    def flush_batch():
        if not pending_texts:
            return
        batch_results = _run_flite_batch(pending_texts)
        all_outputs = []
        for pos_idx, marker in newline_positions:
            all_outputs.append((pos_idx, marker, None))
        for i, (orig, ipa) in enumerate(batch_results):
            all_outputs.append((pending_indices[i], None, (orig, ipa)))
        all_outputs.sort(key=lambda x: x[0])
        for _, marker, result in all_outputs:
            if marker is not None:
                if out_file:
                    out_file.write(marker)
                else:
                    print(marker, end='')
            else:
                orig, ipa = result
                if out_file:
                    out_file.write(ipa)
                    out_file.write(orig)
                else:
                    print((orig, ipa))
        if out_file:
            out_file.flush()
        pending_texts.clear()
        pending_indices.clear()
        newline_positions.clear()

    order_counter = 0
    for i, line in enumerate(lines):
        if i < start_line:
            continue
        normalized_line = normalize(line)
        if fix_line_ends:
            normalized_line = fix_line_ending(normalized_line)
            if normalized_line is None:
                continue
        if normalized_line == "\n":
            newline_positions.append((order_counter, normalized_line))
            order_counter += 1
            continue
        pending_texts.append(normalized_line)
        pending_indices.append(order_counter)
        order_counter += 1
        if len(pending_texts) >= FLITE_BATCH_SIZE:
            flush_batch()
            if checkpoint_path:
                save_checkpoint(checkpoint_path, {
                    "lines_processed": i + 1,
                    "output_bytes": out_file.tell() if out_file else 0,
                    "cached_text": cached_text,
                    "line_end_count": line_end_count,
                    "is_chapter": is_chapter
                })
    if cached_text != "":
        pending_texts.append(cached_text)
        pending_indices.append(order_counter)
        order_counter += 1
    flush_batch()
    if checkpoint_path:
        save_checkpoint(checkpoint_path, {
            "lines_processed": total,
            "output_bytes": out_file.tell() if out_file else 0,
            "cached_text": "",
            "line_end_count": 0,
            "is_chapter": False
        })

PARAGRAPH_PATTERN = re.compile(r'(<p\b[^>]*>)(.*?)(</p>)', re.DOTALL | re.IGNORECASE)
TAG_PATTERN = re.compile(r'<[^>]*>')
SKIP_TAGS = {'script', 'style', 'head', 'noscript', 'svg', 'nav', 'footer'}
SKIP_TAG_PATTERN = re.compile(
    r'<(?P<tag>' + '|'.join(SKIP_TAGS) + r')\b[^>]*>.*?</(?P=tag)>',
    re.DOTALL | re.IGNORECASE
)
SKIP_DIV_IDS = {'secondary', 'actionbar'}
_SKIP_DIV_OPEN = re.compile(
    r'<div\b[^>]*\bid\s*=\s*["\'](' + '|'.join(SKIP_DIV_IDS) + r')["\'][^>]*>',
    re.IGNORECASE
)
_DIV_OPEN = re.compile(r'<div\b', re.IGNORECASE)
_DIV_CLOSE = re.compile(r'</div\s*>', re.IGNORECASE)

def _strip_divs_by_id(content: str) -> str:
    """Remove <div id="secondary"> and <div id="actionbar"> blocks, handling nested divs."""
    while True:
        m = _SKIP_DIV_OPEN.search(content)
        if not m:
            break
        depth = 1
        pos = m.end()
        while depth > 0 and pos < len(content):
            open_m = _DIV_OPEN.search(content, pos)
            close_m = _DIV_CLOSE.search(content, pos)
            if close_m is None:
                break
            if open_m and open_m.start() < close_m.start():
                depth += 1
                pos = open_m.end()
            else:
                depth -= 1
                pos = close_m.end()
        content = content[:m.start()] + content[pos:]
    return content


def _decode_html_text(text: str) -> str:
    decoded = html_module.unescape(text)
    decoded = decoded.replace('\u00a0', ' ')
    decoded = decoded.replace('\u2013', '-')
    decoded = decoded.replace('\u2026', '...')
    return decoded

def _decode_text_nodes(html_str: str) -> str:
    parts = re.split(r'(<[^>]*>)', html_str)
    for i, part in enumerate(parts):
        if not part.startswith('<'):
            parts[i] = _decode_html_text(part)
    return ''.join(parts)

def _prepare_paragraph_texts(match: re.Match):
    open_tag = match.group(1)
    inner = match.group(2)
    close_tag = match.group(3)
    plain_text = TAG_PATTERN.sub('', inner)
    decoded_text = _decode_html_text(plain_text)
    stripped = decoded_text.strip()
    decoded_inner = _decode_text_nodes(inner)
    if not (stripped and any(c.isalpha() for c in stripped)):
        return (open_tag, close_tag, decoded_inner, None, []), []
    parts = re.split(r'(<[^>]*>)', inner)
    flite_needed = []
    for i, part in enumerate(parts):
        if not part.startswith('<'):
            decoded_part = _decode_html_text(part)
            if decoded_part.strip() and any(c.isalpha() for c in decoded_part):
                flite_needed.append((i, normalize(decoded_part), decoded_part))
    return (open_tag, close_tag, decoded_inner, parts, flite_needed), [n for _, n, _ in flite_needed]

def _assemble_paragraph(prep_data, flite_results, paragraph_count, counter):
    open_tag, close_tag, decoded_inner, parts, flite_needed = prep_data
    if parts is None:
        return open_tag + decoded_inner + close_tag
    result_map = {}
    for idx, (part_i, _, decoded_part) in enumerate(flite_needed):
        result_map[part_i] = (decoded_part, flite_results[idx])
    ipa_parts = []
    for i, part in enumerate(parts):
        if part.startswith('<'):
            ipa_parts.append(part)
        elif i in result_map:
            decoded_part, ipa = result_map[i]
            leading = decoded_part[:len(decoded_part) - len(decoded_part.lstrip())]
            trailing = decoded_part[len(decoded_part.rstrip()):]
            ipa_parts.append(leading + ipa.strip() + trailing)
        else:
            ipa_parts.append(_decode_html_text(part))
    ipa_inner = ''.join(ipa_parts)
    if counter % max(1, (paragraph_count // 100)) == 0:
        print(f"paragraph {counter} / {paragraph_count}")
    return open_tag + ipa_inner + close_tag + '\n' + open_tag + decoded_inner + close_tag

def _process_single_paragraph(match: re.Match, paragraph_count: int, counter: int) -> str:
    prep_data, normalized_texts = _prepare_paragraph_texts(match)
    flite_results = []
    for text in normalized_texts:
        _, ipa = run_flite(text)
        flite_results.append(ipa)
    return _assemble_paragraph(prep_data, flite_results, paragraph_count, counter)

def process_html_file(input_path: str, output_path: Optional[str], resume: bool = False):
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    content = SKIP_TAG_PATTERN.sub('', content)
    content = _strip_divs_by_id(content)
    matches = list(PARAGRAPH_PATTERN.finditer(content))
    paragraph_count = len(matches)

    checkpoint_path = get_checkpoint_path(output_path) if output_path else None
    start_paragraph = 0

    if resume and checkpoint_path:
        checkpoint = load_checkpoint(checkpoint_path)
        start_paragraph = checkpoint.get("paragraphs_processed", 0)
        output_bytes = checkpoint.get("output_bytes", 0)
        if start_paragraph > 0:
            print(f"Resuming HTML from paragraph {start_paragraph} / {paragraph_count}")
            if output_bytes > 0 and os.path.exists(output_path):
                with open(output_path, "r+b") as f:
                    f.truncate(output_bytes)

    if output_path:
        mode = "a" if start_paragraph > 0 else "w"
        out_file = open(output_path, mode, encoding='utf-8')
    else:
        out_file = sys.stdout
    prev_end = matches[start_paragraph - 1].end() if start_paragraph > 0 else 0

    for batch_start in range(start_paragraph, len(matches), FLITE_BATCH_SIZE):
        batch_end = min(batch_start + FLITE_BATCH_SIZE, len(matches))
        batch_prep = []
        all_normalized = []
        text_counts = []
        for idx in range(batch_start, batch_end):
            prep_data, normalized_texts = _prepare_paragraph_texts(matches[idx])
            batch_prep.append((idx, prep_data))
            all_normalized.extend(normalized_texts)
            text_counts.append(len(normalized_texts))
        with ThreadPoolExecutor(max_workers=FLITE_MAX_WORKERS) as executor:
            all_ipa_raw = list(executor.map(_call_flite, all_normalized))
        all_flite_results = []
        for raw_ipa, normalized in zip(all_ipa_raw, all_normalized):
            ipa = add_reductions_with_stress(raw_ipa, normalized)
            ipa = add_double_word_reductions(ipa, normalized)
            ipa = handle_t_d(ipa)
            ipa = ipa.replace("ˈ", "")
            all_flite_results.append(ipa)
        result_offset = 0
        for (idx, prep_data), count in zip(batch_prep, text_counts):
            flite_results = all_flite_results[result_offset:result_offset + count]
            result_offset += count
            match = matches[idx]
            out_file.write(content[prev_end:match.start()])
            out_file.write(_assemble_paragraph(prep_data, flite_results, paragraph_count, idx + 1))
            out_file.flush()
            prev_end = match.end()
        if checkpoint_path:
            save_checkpoint(checkpoint_path, {
                "paragraphs_processed": batch_end,
                "output_bytes": out_file.tell()
            })

    out_file.write(content[prev_end:])
    out_file.flush()
    if output_path:
        out_file.close()
    if checkpoint_path:
        remove_checkpoint(checkpoint_path)

def main():
    global cached_text, line_end_count, is_chapter
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str, help="Input text or filename")
    parser.add_argument("-f", "--file", action="store_true",
                        help="Indicate that the input is a filename/dirname instead of text. If dir, will translate all the files in that dir. In this case, output must be given, and be a directory")
    parser.add_argument("-o", "--output", type=str, nargs='?', default=None, help="Optional output file/directory. If not given, will print to stdout")
    parser.add_argument("--html", action="store_true",
                        help="Process an HTML file, running flite only on text content while preserving HTML tags. Decodes HTML entities before processing.")
    parser.add_argument("-r", "--resume", action="store_true",
                        help="Resume from the last checkpoint. Requires --output to be set")

    # Parse the arguments
    args = parser.parse_args()

    if args.resume and not args.output:
        parser.error("--resume requires --output to be set")

    if args.html:
        process_html_file(args.data, args.output, args.resume)
        return
    out_file = None
    if args.file:
        if os.path.isfile(args.data):
            lines = open(args.data).readlines()
        else:
            assert args.output, "When directory is given, output must also be a directory"
            checkpoint_path = get_checkpoint_path(args.output)
            completed_files = set()
            if args.resume:
                checkpoint = load_checkpoint(checkpoint_path)
                completed_files = set(checkpoint.get("completed_files", []))
                if completed_files:
                    print(f"Resuming: skipping {len(completed_files)} already completed files")

            for root, folders, files in os.walk(args.data):
                for file_name in files:
                    input_path = os.path.join(root, file_name)
                    if input_path in completed_files:
                        continue
                    with open(input_path) as f:
                        lines = f.readlines()
                    out_file_name = "ipa_" + file_name
                    with open(os.path.join(args.output, out_file_name), "w") as o:
                        print_ipa(o, lines)
                    completed_files.add(input_path)
                    save_checkpoint(checkpoint_path, {"completed_files": list(completed_files)})

            remove_checkpoint(checkpoint_path)
            return
    else:
        lines = args.data.split("\n")

    if args.output is not None:
        checkpoint_path = get_checkpoint_path(args.output)
        start_line = 0
        if args.resume:
            checkpoint = load_checkpoint(checkpoint_path)
            start_line = checkpoint.get("lines_processed", 0)
            output_bytes = checkpoint.get("output_bytes", 0)
            if start_line > 0:
                print(f"Resuming from line {start_line}")
                cached_text = checkpoint.get("cached_text", "")
                line_end_count = checkpoint.get("line_end_count", 0)
                is_chapter = checkpoint.get("is_chapter", False)
                if output_bytes > 0 and os.path.exists(args.output):
                    with open(args.output, "r+b") as f:
                        f.truncate(output_bytes)
        mode = "a" if start_line > 0 else "w"
        out_file = open(args.output, mode)
        print_ipa(out_file, lines, checkpoint_path=checkpoint_path, start_line=start_line)
        out_file.close()
        remove_checkpoint(checkpoint_path)
    else:
        print_ipa(None, lines)

if __name__ == "__main__":
    main()

    