"""
Property-based tests for api/models.py

Property 6: 非法 voice_id 被拒绝 — Validates: Requirements 5.1
Property 7: 参数范围校验 — Validates: Requirements 5.2, 5.3, 5.4
"""
import pytest
from pydantic import ValidationError
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from api.models import TTSRequest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_VOICE_ID = "test_voice"
VALID_TEXT = "hello"


def _make_request(**kwargs):
    """Build a TTSRequest with sensible defaults, overriding with kwargs."""
    defaults = dict(
        text=VALID_TEXT,
        voice_id=VALID_VOICE_ID,
        num_steps=4,
        t_shift=0.9,
        speed=1.0,
        guidance_scale=3.0,
    )
    defaults.update(kwargs)
    return TTSRequest(**defaults)


# ---------------------------------------------------------------------------
# Property 7: 参数范围校验
# Validates: Requirements 5.2, 5.3, 5.4
# ---------------------------------------------------------------------------

@given(num_steps=st.one_of(st.integers(max_value=0), st.integers(min_value=21)))
@settings(max_examples=50)
def test_property7_num_steps_out_of_range_raises(num_steps):
    """**Validates: Requirements 5.2** — num_steps outside [1, 20] must raise ValidationError."""
    with pytest.raises(ValidationError):
        _make_request(num_steps=num_steps)


@given(t_shift=st.one_of(
    st.floats(max_value=0.09999, allow_nan=False, allow_infinity=False),
    st.floats(min_value=1.00001, allow_nan=False, allow_infinity=False),
))
@settings(max_examples=50)
def test_property7_t_shift_out_of_range_raises(t_shift):
    """**Validates: Requirements 5.3** — t_shift outside [0.1, 1.0] must raise ValidationError."""
    with pytest.raises(ValidationError):
        _make_request(t_shift=t_shift)


@given(speed=st.one_of(
    st.floats(max_value=0.49999, allow_nan=False, allow_infinity=False),
    st.floats(min_value=2.00001, allow_nan=False, allow_infinity=False),
))
@settings(max_examples=50)
def test_property7_speed_out_of_range_raises(speed):
    """**Validates: Requirements 5.4** — speed outside [0.5, 2.0] must raise ValidationError."""
    with pytest.raises(ValidationError):
        _make_request(speed=speed)


@given(guidance_scale=st.one_of(
    st.floats(max_value=0.0, allow_nan=False, allow_infinity=False),
    st.just(0.0),
))
@settings(max_examples=50)
def test_property7_guidance_scale_le_zero_raises(guidance_scale):
    """**Validates: Requirements 5.5** — guidance_scale <= 0 must raise ValidationError."""
    with pytest.raises(ValidationError):
        _make_request(guidance_scale=guidance_scale)


# ---------------------------------------------------------------------------
# Property 6: 非法 voice_id 被拒绝
# Validates: Requirements 5.1
# ---------------------------------------------------------------------------

# Characters that are NOT in [a-zA-Z0-9_-]
_ILLEGAL_CHARS = (
    st.characters(
        blacklist_categories=("Lu", "Ll", "Nd"),  # exclude uppercase, lowercase, digits
        blacklist_characters="_-",
    )
)


@given(
    prefix=st.text(alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="_-"), min_size=0, max_size=10),
    illegal=_ILLEGAL_CHARS,
    suffix=st.text(alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="_-"), min_size=0, max_size=10),
)
@settings(max_examples=100)
def test_property6_illegal_voice_id_raises(prefix, illegal, suffix):
    """**Validates: Requirements 5.1** — voice_id with illegal characters must raise ValidationError."""
    voice_id = prefix + illegal + suffix
    # Ensure the string is non-empty (empty string also fails, but for a different reason)
    assume(len(voice_id) > 0)
    with pytest.raises(ValidationError):
        _make_request(voice_id=voice_id)


# ---------------------------------------------------------------------------
# Sanity: valid inputs should NOT raise
# ---------------------------------------------------------------------------

@given(
    voice_id=st.from_regex(r'^[a-zA-Z0-9_-]{1,20}$', fullmatch=True),
    num_steps=st.integers(min_value=1, max_value=20),
    t_shift=st.floats(min_value=0.1, max_value=1.0, allow_nan=False, allow_infinity=False),
    speed=st.floats(min_value=0.5, max_value=2.0, allow_nan=False, allow_infinity=False),
    guidance_scale=st.floats(min_value=0.001, max_value=100.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=50)
def test_valid_request_does_not_raise(voice_id, num_steps, t_shift, speed, guidance_scale):
    """Valid TTSRequest parameters must not raise ValidationError."""
    req = _make_request(
        voice_id=voice_id,
        num_steps=num_steps,
        t_shift=t_shift,
        speed=speed,
        guidance_scale=guidance_scale,
    )
    assert req.voice_id == voice_id
