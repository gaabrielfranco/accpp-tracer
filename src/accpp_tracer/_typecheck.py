"""Runtime shape checking gate for accpp_tracer.

Shape checks are enabled by default when ``beartype`` is installed.
Set ``ACCPP_TYPECHECK=0`` to disable (e.g. for production inference loops).

To install beartype::

    pip install "accpp-tracer[typecheck]"

Behavior matrix:

+---------------------+------------------+-------------------+
| ACCPP_TYPECHECK env | beartype present | checks active?    |
+=====================+==================+===================+
| "1" (default)       | yes              | yes               |
+---------------------+------------------+-------------------+
| "1" (default)       | no               | no (silent)       |
+---------------------+------------------+-------------------+
| "0"                 | yes / no         | no                |
+---------------------+------------------+-------------------+

The overhead is microseconds per call — negligible compared to GPU
tensor operations.
"""

import os
from typing import Callable, TypeVar

_F = TypeVar("_F", bound=Callable)

_TYPECHECK_ENV = os.environ.get("ACCPP_TYPECHECK", "1").strip()
_REQUESTED = _TYPECHECK_ENV not in ("0", "false", "False", "no", "off")

if _REQUESTED:
    try:
        from beartype import beartype as _beartype
        from jaxtyping import jaxtyped as _jaxtyped
        _ENABLED = True
    except ImportError:
        # beartype not installed — silently disable checks.
        # To enable: pip install "accpp-tracer[typecheck]"
        _ENABLED = False
else:
    _ENABLED = False


def typechecked(fn: _F) -> _F:
    """Apply ``@jaxtyped(typechecker=beartype)`` if shape checking is enabled.

    When enabled, all ``Float[Tensor, "..."]`` annotations on the decorated
    function are validated at each call using beartype + jaxtyping. Named
    dimensions (e.g. ``"d_model"``) are bound on the first annotated argument
    and enforced consistently across all other arguments and the return value.

    When disabled (ACCPP_TYPECHECK=0 or beartype not installed), returns ``fn``
    unchanged with zero overhead.

    Args:
        fn: Function to decorate.

    Returns:
        Decorated function with runtime shape checking, or ``fn`` unchanged.
    """
    if not _ENABLED:
        return fn
    return _jaxtyped(typechecker=_beartype)(fn)  # type: ignore[return-value]
