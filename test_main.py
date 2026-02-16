import json
import os
import re
import subprocess
import tempfile

import pytest

from main import (
    get_next_char,
    get_prev_char,
    normalize,
    handle_t_d,
    add_reductions_with_stress,
    add_double_word_reductions,
    fix_nn,
    fix_numbers,
    fix_line_ending,
    get_checkpoint_path,
    load_checkpoint,
    save_checkpoint,
    remove_checkpoint,
    _decode_html_text,
    _decode_text_nodes,
    _strip_tags_by_attr,
    process_html_file,
    is_verb_in_sentence,
    ipa_vowels,
    ipa_consonants,
    ipa_letters,
    improved_pronounciations,
    normal_reductions,
    h_reduction,
    double_word_reductions,
)


class TestGetNextChar:
    def test_next_char_same_word(self):
        text = ["hello", "world"]
        assert get_next_char(text, 0, 0) == "e"

    def test_next_char_cross_word(self):
        text = ["ab", "cd"]
        assert get_next_char(text, 0, 1) == "c"

    def test_next_char_end_of_text(self):
        text = ["ab"]
        assert get_next_char(text, 0, 1) == ""

    def test_next_char_empty_next_word(self):
        text = ["ab", ""]
        assert get_next_char(text, 0, 1) == ""


class TestGetPrevChar:
    def test_prev_char_same_word(self):
        text = ["hello", "world"]
        assert get_prev_char(text, 0, 1) == "h"

    def test_prev_char_cross_word(self):
        text = ["ab", "cd"]
        assert get_prev_char(text, 1, 0) == "b"

    def test_prev_char_start_of_text(self):
        text = ["ab"]
        assert get_prev_char(text, 0, 0) == ""

    def test_prev_char_empty_prev_word(self):
        text = ["", "cd"]
        assert get_prev_char(text, 1, 0) == ""


class TestNormalize:
    def test_basic_text(self):
        assert normalize("hello world") == "hello world"

    def test_smart_quotes_replaced(self):
        result = normalize("\u2018hello\u2019")
        assert "'" in result

    def test_smart_double_quotes_replaced(self):
        result = normalize("\u201chello\u201d")
        assert '"' in result

    def test_em_dash_replaced(self):
        result = normalize("word\u2014word")
        assert " - " in result

    def test_unicode_normalization(self):
        result = normalize("caf\u00e9")
        assert all(c in __import__("string").printable for c in result)


class TestHandleTD:
    def test_affrication_tr(self):
        result = handle_t_d("tɹip")
        assert "tʃɹ" in result

    def test_affrication_dr(self):
        result = handle_t_d("dɹɪŋk")
        assert "dʒɹ" in result

    def test_affrication_tj(self):
        result = handle_t_d("tjun")
        assert "tʃj" in result

    def test_affrication_dj(self):
        result = handle_t_d("djuk")
        assert "dʒj" in result

    def test_flap_between_vowels(self):
        result = handle_t_d("bʌtə")
        assert "ɾ" in result

    def test_stressed_syllable_preserved(self):
        result = handle_t_d("əˈtæk")
        assert "t" in result
        assert "ɾ" not in result

    def test_post_nasal_deletion(self):
        result = handle_t_d("wɪntli")
        assert "t" not in result.replace("tʃ", "")

    def test_beginning_of_word_preserved(self):
        result = handle_t_d("tɑp")
        assert result.startswith("t")

    def test_skips_special_names(self):
        assert "ɹænd" in handle_t_d("ɹænd")


class TestAddReductionsWithStress:
    def test_of_reduction_between_consonants(self):
        result = add_reductions_with_stress("lɑt ʌv pɹɑbləmz", "lot of problems")
        assert "ə" in result.split(" ")[1]

    def test_of_not_reduced_word_boundary(self):
        result = add_reductions_with_stress("ə ʌv ə", "a of a")
        assert "ʌv" in result

    def test_normal_reduction_applied(self):
        result = add_reductions_with_stress("fɔɹ ðə pipəl", "for the people")
        assert result.split(" ")[0] == "fɝ"

    def test_last_word_not_reduced(self):
        result = add_reductions_with_stress("aɪ kæn", "i can")
        assert result.split(" ")[-1] == "kæn"

    def test_improved_pronunciation_applied(self):
        result = add_reductions_with_stress("fæməli ɪz", "family is")
        assert "fæmli" in result

    def test_h_reduction_after_consonant(self):
        result = add_reductions_with_stress("tɛl hɪm naʊ", "tell him now")
        assert "ɪm" in result


class TestAddDoubleWordReductions:
    def test_want_to_reduction(self):
        result = add_double_word_reductions("wɑnt tu gəʊ", "want to go")
        assert "wɑnə" in result

    def test_do_you_reduction(self):
        result = add_double_word_reductions("du ju noʊ", "do you know")
        assert "dju" in result

    def test_no_reduction_at_end(self):
        result = add_double_word_reductions("wɑnt tu", "want to")
        assert "wɑnə" not in result

    def test_will_not_reduced_before_not(self):
        result = add_double_word_reductions("aɪ wɪl nɑt", "i will not")
        assert "əl" not in result.split(" ")[0]


class TestFixNn:
    def test_nn_word_replaced(self):
        result = fix_nn("the stonn was loud")
        assert "storm" in " ".join(result)

    def test_non_nn_word_unchanged(self):
        result = fix_nn("running is fun")
        assert "running" in result

    def test_m_word_replaced(self):
        result = fix_nn("the stem broke")
        assert any("stern" in w for w in result)


class TestFixNumbers:
    def test_1_becomes_I(self):
        arr = ["hello", "1", "said"]
        result = fix_numbers(arr)
        assert result[1] == "I"

    def test_chapter_number_skipped(self):
        arr = ["chapter", "1"]
        result = fix_numbers(arr)
        assert result[1] == "1"

    def test_parenthesized_number_skipped(self):
        arr = ["(1)", "thing"]
        result = fix_numbers(arr)
        assert result[0] == "(1)"


class TestFixLineEnding:
    def setup_method(self):
        import main
        main.cached_text = ""
        main.line_end_count = 0
        main.is_chapter = False

    def test_complete_sentence_returned(self):
        result = fix_line_ending("Hello world.\n")
        assert result is not None
        assert "Hello world." in result

    def test_incomplete_line_cached(self):
        result = fix_line_ending("Hello world")
        assert result is None

    def test_cached_line_flushed_on_complete(self):
        fix_line_ending("Hello")
        result = fix_line_ending(" world.\n")
        assert result is not None
        assert "Hello" in result

    def test_single_newline_returned_first_time(self):
        result = fix_line_ending("\n")
        assert result == "\n"

    def test_consecutive_newlines_skipped(self):
        fix_line_ending("\n")
        result = fix_line_ending("\n")
        assert result is None

    def test_chapter_handling(self):
        result = fix_line_ending("CHAPTER\n")
        assert result is None


class TestCheckpointing:
    def test_get_checkpoint_path_file(self):
        result = get_checkpoint_path("/tmp/output.txt")
        assert result == "/tmp/output.txt.ipa_checkpoint"

    def test_get_checkpoint_path_dir(self, tmp_path):
        result = get_checkpoint_path(str(tmp_path))
        assert result == os.path.join(str(tmp_path), ".ipa_checkpoint")

    def test_save_and_load_checkpoint(self, tmp_path):
        cp_path = str(tmp_path / "checkpoint.json")
        save_checkpoint(cp_path, {"lines_processed": 42})
        data = load_checkpoint(cp_path)
        assert data["lines_processed"] == 42

    def test_load_nonexistent_checkpoint(self):
        data = load_checkpoint("/tmp/nonexistent_checkpoint_file")
        assert data == {}

    def test_remove_checkpoint(self, tmp_path):
        cp_path = str(tmp_path / "checkpoint.json")
        save_checkpoint(cp_path, {"test": True})
        assert os.path.exists(cp_path)
        remove_checkpoint(cp_path)
        assert not os.path.exists(cp_path)

    def test_remove_nonexistent_checkpoint(self):
        remove_checkpoint("/tmp/nonexistent_checkpoint_file_2")


class TestDecodeHtml:
    def test_html_entity_decoded(self):
        assert _decode_html_text("&amp;") == "&"

    def test_nbsp_replaced(self):
        assert _decode_html_text("\u00a0") == " "

    def test_en_dash_replaced(self):
        assert _decode_html_text("\u2013") == "-"

    def test_ellipsis_replaced(self):
        assert _decode_html_text("\u2026") == "..."

    def test_plain_text_unchanged(self):
        assert _decode_html_text("hello") == "hello"


class TestDecodeTextNodes:
    def test_tags_preserved(self):
        result = _decode_text_nodes("<b>hello&amp;world</b>")
        assert "<b>" in result
        assert "</b>" in result
        assert "hello&world" in result

    def test_only_text_decoded(self):
        result = _decode_text_nodes("before<br/>after")
        assert "<br/>" in result


class TestIsVerbInSentence:
    def test_identifies_verb(self):
        assert is_verb_in_sentence("run", "I run every day") is True

    def test_identifies_non_verb(self):
        assert is_verb_in_sentence("dog", "The dog barked") is False

    def test_word_not_found(self):
        assert is_verb_in_sentence("xyz", "The dog barked") is False


class TestConstants:
    def test_ipa_letters_is_combined(self):
        assert ipa_letters == ipa_vowels + ipa_consonants

    def test_improved_pronounciations_non_empty(self):
        assert len(improved_pronounciations) > 0

    def test_normal_reductions_non_empty(self):
        assert len(normal_reductions) > 0

    def test_h_reduction_non_empty(self):
        assert len(h_reduction) > 0

    def test_double_word_reductions_non_empty(self):
        assert len(double_word_reductions) > 0


class TestStripTagsByAttr:
    def test_strips_secondary_div(self):
        html = '<body><div id="secondary"><p>sidebar</p></div><p>main</p></body>'
        result = _strip_tags_by_attr(html)
        assert 'secondary' not in result
        assert 'sidebar' not in result
        assert '<p>main</p>' in result

    def test_strips_actionbar_div(self):
        html = '<body><div id="actionbar"><button>Click</button></div><p>content</p></body>'
        result = _strip_tags_by_attr(html)
        assert 'actionbar' not in result
        assert 'Click' not in result
        assert '<p>content</p>' in result

    def test_handles_nested_divs(self):
        html = '<div id="secondary"><div class="inner"><p>nested</p></div></div><p>kept</p>'
        result = _strip_tags_by_attr(html)
        assert 'secondary' not in result
        assert 'nested' not in result
        assert '<p>kept</p>' in result

    def test_preserves_other_divs(self):
        html = '<div id="primary"><p>main</p></div><div id="secondary"><p>side</p></div>'
        result = _strip_tags_by_attr(html)
        assert '<div id="primary">' in result
        assert '<p>main</p>' in result
        assert 'secondary' not in result

    def test_strips_both_ids(self):
        html = '<div id="secondary">A</div><div id="actionbar">B</div><p>C</p>'
        result = _strip_tags_by_attr(html)
        assert 'secondary' not in result
        assert 'actionbar' not in result
        assert '<p>C</p>' in result

    def test_no_matching_divs_unchanged(self):
        html = '<div id="content"><p>hello</p></div>'
        assert _strip_tags_by_attr(html) == html

    def test_id_not_first_attribute(self):
        html = '<body><div class="updateable widget-area" id="secondary"><p>sidebar</p></div><p>main</p></body>'
        result = _strip_tags_by_attr(html)
        assert 'secondary' not in result
        assert 'sidebar' not in result
        assert '<p>main</p>' in result

    def test_strips_non_div_tag(self):
        html = '<section id="secondary"><p>inside</p></section><p>outside</p>'
        result = _strip_tags_by_attr(html)
        assert 'secondary' not in result
        assert 'inside' not in result
        assert '<p>outside</p>' in result

    def test_custom_rules(self):
        html = '<div class="ads"><p>ad</p></div><p>content</p>'
        result = _strip_tags_by_attr(html, rules=[('class', 'ads')])
        assert 'ads' not in result
        assert '<p>content</p>' in result


def _flite_available():
    flite_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'flite', 'bin', 'flite')
    if not os.path.isfile(flite_path):
        return False
    try:
        subprocess.check_output([flite_path, "-t", "test", "-i"])
        return True
    except (OSError, subprocess.CalledProcessError):
        return False

requires_flite = pytest.mark.skipif(not _flite_available(), reason="flite binary not available")

IPA_CHARS = set(ipa_vowels + ipa_consonants + "ɾˈˌ")


def _extract_text_from_html(html_content):
    return re.sub(r'<[^>]*>', '', html_content)


def _word_overlap_ratio(text_a, text_b):
    words_a = set(re.findall(r'[a-zA-Z]+', text_a.lower()))
    words_b = set(re.findall(r'[a-zA-Z]+', text_b.lower()))
    if not words_a:
        return 0.0
    return len(words_a & words_b) / len(words_a)


@requires_flite
class TestProcessHtmlFileIntegration:

    def test_simple_html_no_exception(self, tmp_path):
        html = "<html><body><p>The cat sat on the mat.</p></body></html>"
        input_file = tmp_path / "input.html"
        output_file = tmp_path / "output.html"
        input_file.write_text(html, encoding="utf-8")
        process_html_file(str(input_file), str(output_file))
        assert output_file.exists()
        result = output_file.read_text(encoding="utf-8")
        assert len(result) > 0

    def test_output_contains_ipa(self, tmp_path):
        html = "<html><body><p>The quick brown fox jumps over the lazy dog.</p></body></html>"
        input_file = tmp_path / "input.html"
        output_file = tmp_path / "output.html"
        input_file.write_text(html, encoding="utf-8")
        process_html_file(str(input_file), str(output_file))
        result = output_file.read_text(encoding="utf-8")
        found_ipa = any(ch in IPA_CHARS for ch in result)
        assert found_ipa, "Output should contain IPA characters"

    def test_output_preserves_original_text(self, tmp_path):
        html = "<html><body><p>Hello world, this is a simple test.</p></body></html>"
        input_file = tmp_path / "input.html"
        output_file = tmp_path / "output.html"
        input_file.write_text(html, encoding="utf-8")
        process_html_file(str(input_file), str(output_file))
        result = output_file.read_text(encoding="utf-8")
        plain_output = _extract_text_from_html(result)
        assert _word_overlap_ratio("Hello world this is a simple test", plain_output) > 0.5

    def test_html_tags_preserved(self, tmp_path):
        html = "<html><body><p>Some <b>bold</b> text here.</p></body></html>"
        input_file = tmp_path / "input.html"
        output_file = tmp_path / "output.html"
        input_file.write_text(html, encoding="utf-8")
        process_html_file(str(input_file), str(output_file))
        result = output_file.read_text(encoding="utf-8")
        assert "<b>" in result
        assert "</b>" in result

    def test_multiple_paragraphs(self, tmp_path):
        html = (
            "<html><body>"
            "<p>First paragraph with some words.</p>"
            "<p>Second paragraph has different words.</p>"
            "<p>Third paragraph is also here.</p>"
            "</body></html>"
        )
        input_file = tmp_path / "input.html"
        output_file = tmp_path / "output.html"
        input_file.write_text(html, encoding="utf-8")
        process_html_file(str(input_file), str(output_file))
        result = output_file.read_text(encoding="utf-8")
        p_tags = re.findall(r'<p\b[^>]*>', result)
        assert len(p_tags) >= 6, "Each input paragraph should produce an IPA paragraph and an original paragraph"

    def test_non_paragraph_content_preserved(self, tmp_path):
        html = "<html><head><title>Test</title></head><body><h1>Title</h1><p>Paragraph text.</p></body></html>"
        input_file = tmp_path / "input.html"
        output_file = tmp_path / "output.html"
        input_file.write_text(html, encoding="utf-8")
        process_html_file(str(input_file), str(output_file))
        result = output_file.read_text(encoding="utf-8")
        assert "<h1>Title</h1>" in result
        assert "<head>" not in result
        assert "<title>" not in result

    def test_empty_paragraph_no_crash(self, tmp_path):
        html = "<html><body><p></p><p>Real content here.</p></body></html>"
        input_file = tmp_path / "input.html"
        output_file = tmp_path / "output.html"
        input_file.write_text(html, encoding="utf-8")
        process_html_file(str(input_file), str(output_file))
        result = output_file.read_text(encoding="utf-8")
        assert len(result) > 0

    def test_html_entities_handled(self, tmp_path):
        html = "<html><body><p>Tom &amp; Jerry went to the park.</p></body></html>"
        input_file = tmp_path / "input.html"
        output_file = tmp_path / "output.html"
        input_file.write_text(html, encoding="utf-8")
        process_html_file(str(input_file), str(output_file))
        result = output_file.read_text(encoding="utf-8")
        found_ipa = any(ch in IPA_CHARS for ch in result)
        assert found_ipa

    def test_stdout_mode_no_exception(self, tmp_path, capsys):
        html = "<html><body><p>Testing stdout output mode.</p></body></html>"
        input_file = tmp_path / "input.html"
        input_file.write_text(html, encoding="utf-8")
        process_html_file(str(input_file), None)
        captured = capsys.readouterr()
        assert len(captured.out) > 0
        found_ipa = any(ch in IPA_CHARS for ch in captured.out)
        assert found_ipa
