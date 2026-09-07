from swingbird.reply_summary import summarize_reply


class FakeLLM:
    def __init__(self, text_response=""):
        self._text_response = text_response
        self.calls: list[list[dict]] = []

    def complete(self, messages):
        self.calls.append(messages)
        return self._text_response


def test_summarize_reply_returns_the_llm_completion():
    llm = FakeLLM(text_response="Fixed the bug and added a regression test.")

    result = summarize_reply(llm, "I found the off-by-one error and fixed it...")

    assert result == "Fixed the bug and added a regression test."


def test_summarize_reply_sends_the_reply_text_as_the_user_message():
    llm = FakeLLM()

    summarize_reply(llm, "here's what I did")

    messages = llm.calls[0]
    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": "here's what I did"}
