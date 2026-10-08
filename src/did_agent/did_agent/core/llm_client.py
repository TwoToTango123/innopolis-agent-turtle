"""Minimal OpenAI-compatible chat client (ai.mai.ru: Qwen / DeepSeek). Standard library only.

The API key comes from the environment (MAI_API_KEY) or a .env file and is never logged.
"""
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

DEFAULT_BASE_URL = 'https://api-ai.mai.ru/v1'


def load_env_file(path: str) -> dict:
    """KEY=VALUE lines; missing file -> {}."""
    out = {}
    try:
        with open(os.path.expanduser(path)) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, v = line.split('=', 1)
                    out[k.strip()] = v.strip().strip('"\'')
    except FileNotFoundError:
        pass
    return out


@dataclass
class LLMReply:
    content: str
    reasoning: str
    latency: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = ''


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None,
                 env_file: str | None = None, timeout: float = 90.0, max_tokens: int = 4000,
                 temperature: float = 0.2, reasoning_effort: str | None = 'low'):
        env = load_env_file(env_file) if env_file else {}
        self.api_key = api_key or os.environ.get('MAI_API_KEY') or env.get('MAI_API_KEY')
        self.base_url = (base_url or os.environ.get('MAI_BASE_URL') or env.get('MAI_BASE_URL') or DEFAULT_BASE_URL).rstrip('/')
        if not self.api_key:
            raise LLMError('no API key: set MAI_API_KEY or put it into .env')
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort

    def __repr__(self) -> str:          # never show the key
        return f'LLMClient(model={self.model!r}, base_url={self.base_url!r})'

    def chat(self, messages: list[dict]) -> LLMReply:
        body = {'model': self.model, 'messages': messages, 'max_tokens': self.max_tokens, 'temperature': self.temperature}
        if self.reasoning_effort:
            body['reasoning_effort'] = self.reasoning_effort
        req = urllib.request.Request(self.base_url + '/chat/completions', json.dumps(body).encode(),
                                     {'Authorization': 'Bearer ' + self.api_key, 'Content-Type': 'application/json'})
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.load(resp)
        except urllib.error.HTTPError as e:
            raise LLMError(f'HTTP {e.code}: {e.read()[:200].decode(errors="replace")}') from None
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise LLMError(f'network error: {e}') from None
        try:
            choice = data['choices'][0]
            msg = choice['message']
        except (KeyError, IndexError, TypeError):
            raise LLMError(f'unexpected response: {str(data)[:200]}') from None
        usage = data.get('usage') or {}
        return LLMReply(content=msg.get('content') or '', reasoning=msg.get('reasoning_content') or msg.get('reasoning') or '',
                        latency=time.time() - t0, prompt_tokens=usage.get('prompt_tokens', 0),
                        completion_tokens=usage.get('completion_tokens', 0), finish_reason=choice.get('finish_reason') or '')
