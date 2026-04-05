"""
Property-based tests for VoiceStore.

**Validates: Requirements 1.3, 2.1, 2.3, 6.2, 6.3**
"""
from unittest.mock import MagicMock

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from api.voice_store import VoiceStore

# ---------------------------------------------------------------------------
# Helpers / strategies
# ---------------------------------------------------------------------------

# Valid voice_id characters: [a-zA-Z0-9_-], at least 1 char
voice_id_strategy = st.text(
    alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"),
    min_size=1,
    max_size=32,
)

label_strategy = st.text(min_size=0, max_size=64)

encode_dict_strategy = st.fixed_dictionaries({
    "prompt_tokens": st.binary(min_size=1, max_size=16),
    "prompt_features_lens": st.integers(min_value=1, max_value=100),
    "prompt_features": st.binary(min_size=1, max_size=16),
    "prompt_rms": st.floats(min_value=0.0001, max_value=0.1, allow_nan=False),
})


def make_store() -> VoiceStore:
    return VoiceStore(tts=None)


# ---------------------------------------------------------------------------
# Property 1 — voices total 字段不变量
# Validates: Requirements 1.3
# ---------------------------------------------------------------------------

@given(
    entries=st.lists(
        st.tuples(voice_id_strategy, label_strategy, encode_dict_strategy),
        min_size=0,
        max_size=20,
    )
)
@settings(max_examples=100)
def test_property1_voices_total_invariant(entries):
    """**Validates: Requirements 1.3**

    After registering N distinct voice_ids, len(list_voices()) == N.
    Duplicate voice_ids count as one (overwrite semantics).
    """
    store = make_store()
    unique_ids = {}
    for voice_id, label, encode_dict in entries:
        store.register_clone(voice_id, encode_dict, label)
        unique_ids[voice_id] = True

    voices = store.list_voices()
    assert len(voices) == len(unique_ids)


# ---------------------------------------------------------------------------
# Property 2 — 克隆后音色可查询
# Validates: Requirements 2.1, 2.3
# ---------------------------------------------------------------------------

@given(
    voice_id=voice_id_strategy,
    label=label_strategy,
    encode_dict=encode_dict_strategy,
)
@settings(max_examples=100)
def test_property2_clone_voice_queryable(voice_id, label, encode_dict):
    """**Validates: Requirements 2.1, 2.3**

    After registering a voice_id, has_voice() returns True and
    list_voices() contains exactly one entry with that voice_id.
    """
    store = make_store()
    store.register_clone(voice_id, encode_dict, label)

    assert store.has_voice(voice_id)

    voices = store.list_voices()
    matching = [v for v in voices if v.voice_id == voice_id]
    assert len(matching) == 1


# ---------------------------------------------------------------------------
# Property 3 — 重复克隆幂等性
# Validates: Requirements 2.3
# ---------------------------------------------------------------------------

@given(
    voice_id=voice_id_strategy,
    label1=label_strategy,
    label2=label_strategy,
    encode_dict1=encode_dict_strategy,
    encode_dict2=encode_dict_strategy,
)
@settings(max_examples=100)
def test_property3_duplicate_clone_idempotent(voice_id, label1, label2, encode_dict1, encode_dict2):
    """**Validates: Requirements 2.3**

    Registering the same voice_id twice results in exactly one entry
    in list_voices() (overwrite, no duplicates).
    """
    store = make_store()
    store.register_clone(voice_id, encode_dict1, label1)
    store.register_clone(voice_id, encode_dict2, label2)

    voices = store.list_voices()
    matching = [v for v in voices if v.voice_id == voice_id]
    assert len(matching) == 1


# ---------------------------------------------------------------------------
# Property 8 — preset 音色懒加载缓存
# Validates: Requirements 6.2, 6.3
# ---------------------------------------------------------------------------

@given(
    voice_id=voice_id_strategy,
    label=label_strategy,
)
@settings(max_examples=50)
def test_property8_preset_lazy_load_cached(voice_id, label):
    """**Validates: Requirements 6.2, 6.3**

    Calling get_encode_dict() twice on a preset voice invokes
    tts.encode_prompt() exactly once (result is cached).
    """
    mock_tts = MagicMock()
    mock_encode_dict = {
        "prompt_tokens": b"tok",
        "prompt_features_lens": 10,
        "prompt_features": b"feat",
        "prompt_rms": 0.001,
    }
    mock_tts.encode_prompt.return_value = mock_encode_dict

    store = VoiceStore(tts=mock_tts)
    store.register_preset(voice_id, "/fake/path.wav", label)

    result1 = store.get_encode_dict(voice_id)
    result2 = store.get_encode_dict(voice_id)

    # encode_prompt called exactly once
    mock_tts.encode_prompt.assert_called_once_with("/fake/path.wav")
    # both calls return the same cached object
    assert result1 is result2
    assert result1 == mock_encode_dict
