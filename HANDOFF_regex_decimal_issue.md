# Handoff: trailing-zero handling in metric regex parsing

## Context
This issue is in `scripts/collect_iterres.py`, in the metric extraction path for precision/recall/F1 values parsed from logs.

Relevant area:
- `extract_metrics_from_log()` parses lines like:
  - `Baseline (threshold=0.5) -> Precision: 0.6471, Recall: 0.5570, F1: 0.5986`
  - `Best threshold by F1 -> threshold=0.600, Precision=0.8077, Recall=0.5316, F1=0.6412`
- Parsed values are then passed through `correct_deci_number()`.

## Problem
The current regex plus float-correction flow has a trailing-zero issue:

- `0.5570` is parsed as a number/string that becomes `0.557` after `float(...)`
- `0.0417` may be extracted in a way that loses leading/trailing zero context depending on the exact regex capture and later conversion
- Once converted to `float`, trailing zeros are not preserved

So the bug is not only in `correct_deci_number()`; it is also in the extraction and type conversion path.

## Current behavior
`correct_deci_number()` currently:
- converts the input to `float`
- returns it directly if it is `<= 1`
- only applies division logic if the value is greater than 1

That means it cannot preserve formatting like `0.5570` as text, and it is not a reliable recovery method if the regex already lost leading decimal context.

## Why this matters
Metric logs often contain values with meaningful display precision:
- `0.5570` should probably remain `0.5570` if the output is meant to mirror the log
- `0.0417` should remain `0.0417`

If the data is later written to CSV or plotted, the distinction between `0.557` and `0.5570` may matter for presentation even if the numeric value is the same.

## Likely root cause
There are two separate issues:

1. **Regex capture is numeric-only**
   - Patterns like `([\d.]+)` capture digits and dots only
   - This is fine for numeric parsing, but not for preserving exact formatting

2. **`float()` removes trailing zeros**
   - `float("0.5570")` becomes `0.557`
   - `float("0.0417")` becomes `0.0417`

## Suggested next steps
Choose one of these approaches:

### Option A: Preserve display formatting
- Keep the original matched string for the metric fields
- Only convert to numeric later, when needed for computation/plotting
- Use a separate formatted string field if exact log representation matters

### Option B: Use `Decimal` for parsing
- Parse metric strings with `Decimal` instead of `float`
- This preserves scale better than float
- Still note that many downstream libraries will convert back to float for plotting

### Option C: Improve extraction logic first
- Make regex capture the exact metric segment more explicitly
- Ensure the match groups do not over-trim or require later correction
- Example goal: parse `Recall: 0.5570` as a clean, direct numeric string without additional heuristics

## Files to check
- `scripts/collect_iterres.py`
  - `correct_deci_number()`
  - `extract_metrics_from_log()`
  - baseline and best-threshold regex blocks

## Important note
If the goal is strictly numeric analysis, trailing zero loss is harmless.
If the goal is to preserve the original log precision for reporting, `float()` is the wrong final storage type.

## Status at handoff
- Regexes for baseline/best-threshold parsing were already tightened up.
- The remaining issue is how decimals are normalized and stored after regex extraction.
- The next session should decide whether to preserve original metric strings or keep numeric values only.
