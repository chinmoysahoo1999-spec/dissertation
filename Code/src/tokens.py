"""
tokens.py — Block 3: token-level generation helpers.

The single public function is :func:`find_first_and_next_token`. Logic is
preserved exactly from the original notebook; only the surface has changed
(tokenizer is now an explicit argument).
"""

from __future__ import annotations

from typing import List, Optional


def find_first_and_next_token(
    text: str,
    entity: str,
    idx: int,
    input_id: List[List[int]],
    tokenizer,
    prompt: str = "",
) -> Optional[List]:
    """
    Locate the *first token of the entity* and *the next token after the entity*
    in the tokenizer's output.

    The trick from the original code: we insert a ``@`` marker right after
    the entity, retokenize, and read off the first token at the entity's
    position and the token after the ``@``.

    Returns
    -------
    [first_token, next_token, entity_len, last_id]  on success
    None                                            on any structural mismatch
    """
    new_text = (
        f"{text[:idx].strip()} "
        f"{text[idx:].replace(entity, entity + ' @', 1).strip()}"
    )
    new_input_id = tokenizer(
        prompt + new_text.strip(), return_tensors="pt"
    )["input_ids"].tolist()[0]

    # Verify the prompt prefix matches token-for-token; otherwise we're not
    # comparing apples to apples.
    for i in range(len(input_id[0])):
        if input_id[0][i] != new_input_id[i]:
            return None

    first_token = new_input_id[len(input_id[0])]

    # The @ may tokenize as either "@" or " @"; try both.
    at_candidates = list(tokenizer("@", add_special_tokens=False)["input_ids"])
    at_candidates += list(tokenizer(" @", add_special_tokens=False)["input_ids"])

    at_position = None
    for at_tok in at_candidates:
        try:
            at_position = new_input_id.index(at_tok, len(input_id[0]))
            break
        except ValueError:
            continue

    if at_position is None or at_position >= len(new_input_id) - 1:
        return None

    next_token = new_input_id[at_position + 1]
    entity_len = at_position - len(input_id[0])
    last_id = new_input_id[at_position + 1 :]

    return [first_token, next_token, entity_len, last_id]
