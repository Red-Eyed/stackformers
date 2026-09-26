# Result records

`Result[T, E]` is the closed union `Ok[T] | Err[E]`. Both alternatives are immutable generic
NamedTuples: success carries `value`, failure carries `error`. Their distinct literal `kind`
tags distinguish opposite variants when their payloads compare equal. Payloads remain native values;
records do not copy tensors, detach gradients, or discard exception causes and tracebacks.

This module is a pure data contract with no I/O, exception policy, or dependencies outside the
standard library. Handle both variants with pattern matching and end closed-union dispatch
with `assert_never`. There are no unchecked unwrap methods or functional composition helpers.
Type annotations constrain checked callers; these records do not validate untyped inputs.
Use keyword class patterns as below: PyTorch 2.11 Dynamo rejects positional class patterns
for these generic NamedTuples during strict export and full-graph compilation. Keyword patterns
avoid that unsupported operation without disabling tracing or adding graph breaks.

```python
from typing import assert_never

from stackformers.result import Err, Ok, Result


def describe(outcome: Result[int, str]) -> str:
    """Handle successful integers and expected rejection reasons explicitly."""
    match outcome:
        case Ok(value=value):
            return f"accepted {value}"
        case Err(error=error):
            return f"rejected: {error}"
        case _:
            assert_never(outcome)
```

The outer caller chooses recovery, logging, or termination. Stackformers' existing public
adapters keep their tensor/configuration and exception contracts; the raising adapter lives
separately in `_result.py`. Add domain error records when a new operation needs richer failure
detail rather than extending the success/failure container.
