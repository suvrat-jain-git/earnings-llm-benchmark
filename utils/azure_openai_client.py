"""
Azure OpenAI client wrapper with retry logic and cost tracking.
"""
import json
import time as time_module
from pathlib import Path
from openai import AzureOpenAI
from dotenv import dotenv_values

import config


class AzureGPTClient:
    """Wrapper for Azure OpenAI GPT calls with retry and cost tracking."""

    def __init__(self, env_path: str | Path = None):
        env_path = env_path or config.ENV_FILE_PATH
        env = dotenv_values(str(env_path))

        self.client = AzureOpenAI(
            api_key=env["AZURE_OPENAI_API_KEY"],
            azure_endpoint=env["AZURE_OPENAI_ENDPOINT"],
            api_version=env.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        )
        self.deployment = env.get("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4.1")

        # Cost tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.call_log = []
        self.log_path = config.RESULTS_DIR / "gpt_raw_responses.jsonl"

    def call(self, system_prompt: str, user_prompt: str,
             temperature: float = None, max_tokens: int = None,
             experiment_name: str = "", ticker: str = "") -> dict:
        """
        Call GPT with retry logic.

        Returns:
            {
                "content": str (response text),
                "input_tokens": int,
                "output_tokens": int,
                "total_tokens": int,
            }
        """
        temperature = temperature if temperature is not None else config.GPT_TEMPERATURE
        max_tokens = max_tokens or config.GPT_MAX_TOKENS

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        last_error = None
        for attempt in range(config.API_RETRY_MAX):
            try:
                response = self.client.chat.completions.create(
                    model=self.deployment,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                content = response.choices[0].message.content.strip()
                usage = response.usage
                result = {
                    "content": content,
                    "input_tokens": usage.prompt_tokens,
                    "output_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                }

                # Track costs
                self.total_input_tokens += usage.prompt_tokens
                self.total_output_tokens += usage.completion_tokens

                # Log raw response
                log_entry = {
                    "experiment": experiment_name,
                    "ticker": ticker,
                    "input_tokens": usage.prompt_tokens,
                    "output_tokens": usage.completion_tokens,
                    "response": content,
                }
                self.call_log.append(log_entry)
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_entry) + "\n")

                return result

            except Exception as e:
                last_error = e
                if attempt < config.API_RETRY_MAX - 1:
                    delay = config.API_RETRY_BASE_DELAY * (2 ** attempt)
                    print(f"  API error (attempt {attempt+1}): {e}. Retrying in {delay:.0f}s...")
                    time_module.sleep(delay)

        raise RuntimeError(f"GPT call failed after {config.API_RETRY_MAX} attempts: {last_error}")

    def get_cost_summary(self) -> dict:
        """Return cost tracking summary."""
        # Rough pricing estimate (adjust to actual Azure pricing)
        input_cost_per_1m = 2.50   # USD per 1M input tokens
        output_cost_per_1m = 10.00  # USD per 1M output tokens

        input_cost = self.total_input_tokens / 1_000_000 * input_cost_per_1m
        output_cost = self.total_output_tokens / 1_000_000 * output_cost_per_1m

        return {
            "total_calls": len(self.call_log),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "estimated_input_cost_usd": round(input_cost, 4),
            "estimated_output_cost_usd": round(output_cost, 4),
            "estimated_total_cost_usd": round(input_cost + output_cost, 4),
        }


