from llama_benchy.prompts import PromptGenerator, TASK_INSTRUCTIONS, TASK_SYSTEM_PROMPT


class _IdentityTokenizer:
    """A tokenizer stand-in where each character is one token, so slicing is trivial."""

    def encode(self, text, add_special_tokens=False):
        return list(text)

    def decode(self, token_ids):
        return "".join(token_ids)


class _FakeCorpus:
    def __init__(self, text, name="book"):
        self._tokenizer = _IdentityTokenizer()
        self._tokens = list(text)
        self._name = name

    def get_tokenizer(self):
        return self._tokenizer

    def get_tokens(self):
        return self._tokens

    def get_name(self):
        return self._name


def _make_generator(text="abcdefghijklmnopqrstuvwxyz" * 10, name="my_book"):
    return PromptGenerator(_FakeCorpus(text, name=name))


def test_continue_mode_unchanged_when_task_mode_absent():
    gen = _make_generator()
    context, prompt = gen.generate(prompt_tokens=10, context_tokens=5, mode="continue")

    assert TASK_SYSTEM_PROMPT not in context
    assert "```" not in prompt
    assert len(context) == 5
    assert len(prompt) == 10


def test_task_mode_builds_chat_shaped_prompt_with_file_excerpt():
    gen = _make_generator(name="Sherlock Holmes")
    context, prompt = gen.generate(prompt_tokens=12, context_tokens=8, mode="task")

    assert context.startswith(TASK_SYSTEM_PROMPT)
    # Depth slice (8 tokens) is appended after the fixed system prompt.
    assert context == f"{TASK_SYSTEM_PROMPT}\n\n" + context[-8:]
    assert len(context) - len(TASK_SYSTEM_PROMPT) - 2 == 8

    assert prompt.startswith("Here is Sherlock Holmes (excerpt):\n```\n")
    assert "```" in prompt
    assert any(instr in prompt for instr in TASK_INSTRUCTIONS)


def test_task_mode_with_zero_depth_has_no_extra_context_block():
    gen = _make_generator()
    context, _ = gen.generate(prompt_tokens=10, context_tokens=0, mode="task")

    assert context == TASK_SYSTEM_PROMPT


def test_task_mode_instruction_choice_is_deterministic_across_generators():
    gen_a = _make_generator()
    gen_b = _make_generator()

    _, prompt_a = gen_a.generate(prompt_tokens=10, context_tokens=0, mode="task")
    _, prompt_b = gen_b.generate(prompt_tokens=10, context_tokens=0, mode="task")

    instr_a = next(instr for instr in TASK_INSTRUCTIONS if instr in prompt_a)
    instr_b = next(instr for instr in TASK_INSTRUCTIONS if instr in prompt_b)
    assert instr_a == instr_b


def test_task_mode_pp_slice_sized_same_as_continue_mode():
    # The excerpt shown to the model should be sized exactly by prompt_tokens,
    # same mechanism as --prompt-mode continue.
    gen = _make_generator()
    _, prompt = gen.generate(prompt_tokens=15, context_tokens=3, mode="task")

    excerpt = prompt.split("```\n", 1)[1].rsplit("\n```\n", 1)[0]
    assert len(excerpt) == 15


def test_task_mode_no_cache_appends_suffix_for_uniqueness():
    gen = _make_generator()
    _, prompt = gen.generate(prompt_tokens=10, context_tokens=0, mode="task", no_cache=True)

    # uuid4() renders as 32 hex digits split by 4 dashes.
    assert prompt.strip().rsplit(" ", 1)[-1].count("-") == 4


def test_generate_batch_reuses_generate_for_each_member():
    gen = _make_generator()
    batch = gen.generate_batch(3, prompt_tokens=6, context_tokens=4, mode="task")

    assert len(batch) == 3
    for context, prompt in batch:
        assert context.startswith(TASK_SYSTEM_PROMPT)
        assert prompt.startswith("Here is ")
