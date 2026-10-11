import os
import unittest
from unittest.mock import patch

from commaqa.models.gpt3generator import get_openai_client


class QwenOpenAIClientTest(unittest.TestCase):
    def test_local_llm_address_becomes_openai_compatible_base_url(self):
        with patch.dict(
            os.environ,
            {"LLM_SERVER_HOST": "http://localhost", "LLM_SERVER_PORT": "8010"},
            clear=True,
        ):
            client = get_openai_client()

        self.assertEqual(str(client.base_url), "http://localhost:8010/v1/")
        self.assertEqual(client.api_key, "EMPTY")


if __name__ == "__main__":
    unittest.main()
