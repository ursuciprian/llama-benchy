import random
import uuid
import numpy as np
from typing import Tuple, List

from .corpus import TokenizedCorpus

# Fixed short system prompt for --prompt-mode task. Kept short and constant so
# it never affects the pp/depth token budgets that size the corpus slices.
TASK_SYSTEM_PROMPT = "You are a coding assistant. Answer with code only."

# ~12 coding instructions that push the model toward long code answers,
# so decode looks like real agent traffic instead of raw-text continuation.
TASK_INSTRUCTIONS = [
    "Add unit tests for the functions above.",
    "Refactor the longest function into smaller ones and return the full new code.",
    "Port this excerpt to Rust.",
    "Add doc comments to every function and return the complete excerpt.",
    "Add type hints throughout and return the complete file.",
    "Rewrite this excerpt to use async/await and return the complete file.",
    "Add comprehensive error handling and return the complete file.",
    "Convert this excerpt to Go and preserve behavior.",
    "Add logging statements to every function and return the complete file.",
    "Optimize the slowest function for performance and return the complete file.",
    "Add input validation to every public function and return the complete file.",
    "Write a CLI wrapper around the functions above and return the complete file.",
]

# Fixed seed so instruction choice is reproducible across runs (deterministic,
# not random per-invocation), while still varying across a batch/suite.
_TASK_INSTRUCTION_SEED = 1337


class PromptGenerator:
    def __init__(self, corpus: TokenizedCorpus):
        self.corpus = corpus
        self.tokenizer = corpus.get_tokenizer()
        self.all_tokens = corpus.get_tokens()
        self._task_rng = random.Random(_TASK_INSTRUCTION_SEED)

    def _next_task_instruction(self) -> str:
        return self._task_rng.choice(TASK_INSTRUCTIONS)

    def _build_task_prompt(self, depth_text: str, excerpt_text: str) -> Tuple[str, str]:
        """
        Builds a chat-shaped (system, user) pair for an agent-like coding turn.

        - System message: short fixed instruction, always present.
        - Depth slice (previous "conversation" tokens): kept as an additional
          cached system-role suffix, exactly the same slice/mechanism as
          --prompt-mode continue, so prefix-cache behavior across the
          concurrency batch (and the --enable-prefix-caching context-load
          phase) is unchanged.
        - Prompt/pp slice: sized exactly like today, presented as a file
          excerpt in the user message, followed by one deterministic
          instruction.
        """
        system_text = TASK_SYSTEM_PROMPT
        if depth_text:
            system_text = f"{TASK_SYSTEM_PROMPT}\n\n{depth_text}"

        name = self.corpus.get_name()
        instruction = self._next_task_instruction()
        user_text = f"Here is {name} (excerpt):\n```\n{excerpt_text}\n```\n{instruction}"

        return system_text, user_text

    def generate(self, prompt_tokens: int, context_tokens: int = 0, no_cache: bool = False, mode: str = "continue") -> Tuple[str, str]:
        """
        Generates a single (context, prompt) pair.
        """
        suffix = ""
        suffix_len = 0
        if no_cache:
            suffix = f" {uuid.uuid4()}"
            suffix_len = len(self.tokenizer.encode(suffix, add_special_tokens=False))

        # Adjust prompt tokens to fetch from text
        text_prompt_tokens = max(0, prompt_tokens - suffix_len)

        # Create a pool of tokens large enough
        total_needed = text_prompt_tokens + context_tokens

        # Create a local reference to tokens to potentially extend
        current_tokens = self.all_tokens

        if len(current_tokens) < total_needed:
            # Repeat tokens if not enough
            current_tokens = current_tokens * (total_needed // len(current_tokens) + 2)

        # Pick a random start position
        max_start = len(current_tokens) - total_needed
        start_idx = np.random.randint(0, max_start)

        selected_tokens = current_tokens[start_idx : start_idx + total_needed]

        context_text = self.tokenizer.decode(selected_tokens[:context_tokens]) if context_tokens > 0 else ""
        prompt_text = self.tokenizer.decode(selected_tokens[context_tokens:])

        if mode == "task":
            context_text, prompt_text = self._build_task_prompt(context_text, prompt_text)
            if no_cache:
                prompt_text += suffix
            return context_text, prompt_text

        if no_cache:
            prompt_text += suffix

        return context_text, prompt_text

    def generate_batch(self, batch_size: int, prompt_tokens: int, context_tokens: int = 0, no_cache: bool = False, mode: str = "continue") -> List[Tuple[str, str]]:
        """
        Generates a batch of (context, prompt) pairs.
        """
        return [self.generate(prompt_tokens, context_tokens, no_cache, mode) for _ in range(batch_size)]
