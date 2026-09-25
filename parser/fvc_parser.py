"""Reduce a model response to a vegetation percentage and a confidence value.

This is the parser behind every published estimate. A response is free text: a
model may return a bare JSON object, wrap it in a code fence, or narrate its
reasoning and restate the requested output format before answering. The parser
finds the answer in all three cases, and records the response as missing when it
carries no answer.

Two rules carry the work.

**Take the last well-formed object, not the first.** A model that narrates often
restates the requested output format (`{"vegetation_percent": <number>, ...}`)
partway through its reasoning, or gives an intermediate figure before revising
it. Its answer is therefore the last JSON object in the response, so candidates
are tried from the end backwards.

**Let a JSON decoder say where each object begins and ends.** `raw_decode` is
applied at every `{` and reports whether a complete object starts there. Pairing
each `{` with a later `}` instead would miscount any brace written inside a
string value, which happens whenever a model quotes the requested output format
in its prose, and would cut the object at the wrong character.

A response with no complete object is recorded as missing. Nothing is
reconstructed from partial output, and no number is inferred from prose.
"""

import json
import re

# A fence may open with ```json or plain ```; both are stripped before the
# whole-string attempt, which is the common case and needs no extraction.
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
_DECODER = json.JSONDecoder()

# Three responses of 122,136 wrote the cover under "vegetation_cover" rather
# than the requested "vegetation_percent". The alias is listed explicitly so
# the accepted vocabulary stays visible and closed.
_COVER_KEYS = ("vegetation_percent", "vegetation_cover")


def extract_json_objects(text):
    """Every complete JSON object in `text`, in order of appearance.

    Tries to decode at each `{`. Positions that do not begin a valid object are
    skipped, so prose surrounding the answer costs nothing but a failed decode.
    """
    out = []
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _end = _DECODER.raw_decode(text, i)
        except ValueError:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def _pair(obj):
    """(cover, confidence) if the object carries both as numbers, else None.

    `bool` is excluded deliberately: it is a subclass of `int` in Python, so a
    response of `{"vegetation_percent": true}` would otherwise parse as 1.0.
    """
    veg = next((obj[k] for k in _COVER_KEYS if k in obj), None)
    conf = obj.get("confidence")

    def is_num(x):
        return isinstance(x, (int, float)) and not isinstance(x, bool)

    return (float(veg), float(conf)) if is_num(veg) and is_num(conf) else None


def parse_response(raw_text):
    """(cover, confidence) for a response, or None if it carries no answer.

    Confidence is returned exactly as the model wrote it. The scale varies
    between models and prompts (0.8, 80 and 8 all occur), and normalising it
    requires context this function does not have, so that step is left to the
    caller.
    """
    text = _FENCE_RE.sub("", raw_text.strip()).strip()
    try:
        got = _pair(json.loads(text))
        if got:
            return got
    except (json.JSONDecodeError, ValueError):
        pass

    for obj in reversed(extract_json_objects(raw_text)):
        got = _pair(obj)
        if got:
            return got
    return None
