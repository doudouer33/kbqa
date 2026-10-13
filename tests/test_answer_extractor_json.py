import contextlib
import io
import json
import os
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("OPENAI_API_KEY", "EMPTY")

from commaqa.inference.data_instances import QuestionAnsweringStep, QuestionGenerationStep, StructuredDataInstance
from commaqa.inference.ircot import AnswerExtractor
from commaqa.inference.model_search import QuestionSearchBase, SearchState
from commaqa.inference.participant_qa import LLMQAParticipantModel


REGEX = r".* answer is:? (.*)\.?"
ANSWERS = (
    "Atlético Mineiro",
    "4 × 100 metre relay",
    "Beyoncé",
    "中文 한국어 العربية 😀 𝄞 e\u0301",
    "plain ASCII",
    "yes",
    'He said "hello" to Beyoncé',
    r"literal \u00e9 and \n text",
    r"C:\Users\Beyoncé\new\test.txt",
    r"\\server\share\folder",
)


def state_with_text(text, source="last_answer"):
    data = StructuredDataInstance({"qid": "test", "question": "Question?", "titles": [], "paras": [], "metadata": {}})
    data.add_qgen(
        QuestionGenerationStep(question=text if source == "last_question" else "Question?", score=0, participant="copy")
    )
    if source == "last_answer":
        data.add_answer(QuestionAnsweringStep(answer=text, score=0, participant="llmqa"))
    return SearchState(data, "extract_answer")


def extractor(**kwargs):
    settings = dict(regex=REGEX, query_source="last_answer", match_all_on_failure=True, remove_last_fullstop=True)
    settings.update(kwargs)
    return AnswerExtractor(**settings)


def final_prediction(state):
    search = QuestionSearchBase(None)
    with patch.object(search, "find_answer_decomp", return_value=(state, [])):
        return search.return_qid_prediction({"qid": "test", "question": "Question?"}, silent=True)[1]


def participant_for_raw(raw):
    generator = Mock()
    generator.generate_text_sequence.return_value = [(raw, 0)]
    with patch("commaqa.execution.llm_qa_model.GPT3Generator", return_value=generator):
        participant = LLMQAParticipantModel(next_model="extract_answer", gen_model="gpt3")
    participant.qa_model.prompt = ""
    return participant


class AnswerExtractorJSONTest(unittest.TestCase):
    def test_participant_serialization_roundtrip(self):
        participant = participant_for_raw("unused")
        for answer in ANSWERS + ("line one\nline two\tend\r", ["Beyoncé", r"\u00e9"], {"name": "中文"}, 42, True, None):
            with self.subTest(answer=answer):
                state = state_with_text("unused")
                result = participant.update_state(answer, state)
                self.assertEqual(result.data.get_last_answer(), json.dumps(answer))
                self.assertEqual(json.loads(result.data.get_last_answer()), answer)
                self.assertEqual(state.data.get_last_answer(), "unused")

    def test_serialized_cot_through_actual_participant_extractor_and_final_prediction(self):
        for answer in ANSWERS:
            with self.subTest(answer=answer):
                participant = participant_for_raw("Reasoning. So the answer is: " + answer + ".")
                generated = participant.query(state_with_text("unused"))
                result = extractor().query(generated)
                self.assertEqual(result.data.get_last_answer(), answer)
                self.assertEqual(final_prediction(result), answer)
                self.assertEqual(json.loads(json.dumps(final_prediction(result))), answer)
                participant.qa_model.generator.generate_text_sequence.assert_called_once()

    def test_json_strings_decode_once_before_matching(self):
        for ensure_ascii in (True, False):
            for answer in ANSWERS:
                with self.subTest(ensure_ascii=ensure_ascii, answer=answer):
                    serialized = json.dumps("So the answer is: " + answer + ".", ensure_ascii=ensure_ascii)
                    self.assertEqual(extractor().query(state_with_text(serialized)).last_output, answer)

    def test_plain_strings_and_last_question_preserve_literal_escapes(self):
        for source in ("last_answer", "last_question"):
            for answer in ANSWERS:
                with self.subTest(source=source, answer=answer):
                    result = extractor(query_source=source).query(
                        state_with_text("So the answer is: " + answer + ".", source)
                    )
                    self.assertEqual(result.last_output, answer)

    def test_last_question_keeps_legacy_outer_quote_handling(self):
        question = r'"So the answer is: literal \u00e9."'
        result = extractor(query_source="last_question").query(state_with_text(question, "last_question"))
        self.assertEqual(result.last_output, r"literal \u00e9")

    def test_invalid_quoted_json_keeps_legacy_outer_quote_handling(self):
        result = extractor().query(state_with_text(r'"So the answer is: C:\Users\new."'))
        self.assertEqual(result.last_output, r"C:\Users\new")

    def test_json_escaped_newline_is_restored_before_regex(self):
        # A regex spanning lines is only used here to observe the decoded text.
        # The production regex and its newline behavior are tested separately.
        raw = "So the answer is: line one\nline two."
        result = extractor(regex=r"(?s).* answer is:? (.*)\.?").query(state_with_text(json.dumps(raw)))
        self.assertEqual(result.last_output, "line one\nline two")
        self.assertEqual(final_prediction(result), "line one\nline two")

    def test_original_regex_and_fallback_remain_line_sensitive(self):
        raw = "Reasoning before newline.\nSo the answer is: yes."
        result = extractor().query(state_with_text(json.dumps(raw)))
        self.assertEqual(result.last_output, "Reasoning before newline")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(extractor(match_all_on_failure=False).query(state_with_text(json.dumps(raw))), [])
        self.assertEqual(extractor().regex.pattern, REGEX)
        self.assertEqual(extractor().regex.flags, 32)  # re.UNICODE; no DOTALL.

    def test_no_match_and_fullstop_settings_are_unchanged(self):
        text = json.dumps("A plain answer.")
        self.assertEqual(extractor().query(state_with_text(text)).last_output, "A plain answer")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(extractor(match_all_on_failure=False).query(state_with_text(text)), [])
        self.assertEqual(
            extractor(remove_last_fullstop=False)
            .query(state_with_text(json.dumps("So the answer is: yes.")))
            .last_output,
            "yes.",
        )

    def test_json_lists_keep_intermediate_representation_and_final_list_semantics(self):
        answer = ["Beyoncé", "4 × 100 metre relay", r"\u00e9", r"C:\new\test"]
        participant = participant_for_raw(json.dumps(answer))
        generated = participant.query(state_with_text("unused"))
        self.assertEqual(generated.data.get_last_answer(), json.dumps(answer))
        result = extractor().query(generated)
        self.assertEqual(result.last_output, json.dumps(answer))
        self.assertEqual(final_prediction(result), answer)

    def test_list_captured_inside_cot_keeps_json_structure(self):
        answer = ["Beyoncé", r"\u00e9", "中文"]
        participant = participant_for_raw("So the answer is: " + json.dumps(answer) + ".")
        result = extractor().query(participant.query(state_with_text("unused")))
        self.assertEqual(result.last_output, json.dumps(answer))
        self.assertEqual(final_prediction(result), answer)

    def test_other_json_values_are_not_coerced(self):
        for answer in ({"name": "Beyoncé"}, 42, True, None):
            with self.subTest(answer=answer):
                text = json.dumps(answer)
                result = extractor().query(state_with_text(text))
                self.assertEqual(result.last_output, text)
                self.assertEqual(final_prediction(result), text)

    def test_literal_escaped_list_is_not_repaired_or_decoded_again(self):
        answer = r"[\"Beyoncé\", \"C:\\new\"]"
        result = extractor().query(state_with_text(json.dumps("So the answer is: " + answer + ".")))
        self.assertEqual(result.last_output, answer)
        self.assertEqual(final_prediction(result), answer)

    def test_direct_qa_retains_existing_final_json_decode(self):
        for answer in ANSWERS + (["Beyoncé", r"\u00e9"],):
            with self.subTest(answer=answer):
                # String output is intentionally JSON encoded by the stub model.
                participant = participant_for_raw(json.dumps(answer))
                participant.next_model = None
                self.assertEqual(final_prediction(participant.query(state_with_text("unused"))), answer)


if __name__ == "__main__":
    unittest.main()
