# API/

Code that ran all six models through the OpenRouter gateway, the hosted-API comparison. It covers
the base photographs only; the two local paths additionally cover the masked-gray and rectified
variants.

## Files

- **`Visual-LLM_Prediction_generation.ipynb`** is the acquisition notebook. Three pieces carry the
  work:
  - `load_and_normalize_image()` and `pil_to_base64_jpeg()` prepare the photograph: open, convert to
    RGB, `thumbnail()` to a 1,536 px maximum side with BICUBIC resampling, re-encode as JPEG at
    quality 90. Base64 is where this path diverges, because the image travels inside the request
    body as a `data:image/jpeg;base64,...` URL, where local inference hands the same preprocessed
    JPEG over by file path.
  - `call_openrouter()` sends the request at temperature 0, with the prompt text before the image in
    the content list. It retries with exponential backoff on rate-limit and server errors, and if a
    response carries no answer it reissues the call once before recording the value as missing.
  - `parse_strict_json()` is the parser published in `../../parser/fvc_parser.py`, imported under
    that name. It is the same parser used on both local paths and behind every reported estimate, so
    a response is read here exactly as it is read there: the last well-formed JSON object carrying a
    vegetation percentage and a confidence value as numbers, with `vegetation_cover` accepted
    alongside `vegetation_percent` for the cover field, and a response with no complete object
    recorded as missing rather than reconstructed from partial output.

## Preprocessing

The same resize and re-encode settings as the local paths: 1,536 px maximum side, BICUBIC, JPEG
quality 90. Only the Base64 transport encoding is specific to this path.
