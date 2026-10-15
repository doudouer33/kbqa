import os
import unittest
import importlib
from unittest.mock import patch

os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
from commaqa.models import gpt3generator


class QwenOpenAIClientTest(unittest.TestCase):
    def test_local_llm_address_becomes_openai_compatible_base_url(self):
        with patch.dict(
            os.environ,
            {"OPENAI_BASE_URL": "http://localhost:8010/v1", "OPENAI_API_KEY": "EMPTY"},
            clear=True,
        ):
            # The verified generator constructs OpenAI() from OPENAI_*;
            # formal transport adaptation is experiment-local.
            client = importlib.reload(gpt3generator).client

        self.assertEqual(str(client.base_url), "http://localhost:8010/v1/")
        self.assertEqual(client.api_key, "EMPTY")


if __name__ == "__main__":
    unittest.main()
