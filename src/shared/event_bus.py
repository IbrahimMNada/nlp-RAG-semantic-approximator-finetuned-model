"""
Simple in-process event bus for module-to-module communication.

Modules register async handlers for named signals. Other modules send requests
through the bus and receive responses directly — no HTTP overhead.
"""
import logging
from typing import Any, Callable, Dict

logger = logging.getLogger(__name__)

# signal_name -> handler callable
_handlers: Dict[str, Callable[..., Any]] = {}


def register(signal_name: str, handler: Callable[..., Any]) -> None:
    """
    Register a handler for a named signal.

    Args:
        signal_name: Unique signal identifier.
        handler: Async or sync callable that will handle the signal.
    """
    if signal_name in _handlers:
        logger.warning(
            "Overwriting existing handler for signal '%s'", signal_name
        )
    _handlers[signal_name] = handler
    logger.info("Handler registered for signal '%s'", signal_name)


async def send(signal_name: str, **kwargs) -> Any:
    """
    Send a signal and return the handler's result.

    Args:
        signal_name: Name of the signal to send.
        **kwargs: Arguments forwarded to the handler.

    Returns:
        The return value of the registered handler.

    Raises:
        RuntimeError: If no handler is registered for the signal.
    """
    handler = _handlers.get(signal_name)

    if handler is None:
        raise RuntimeError(
            f"No handler registered for signal '{signal_name}'. "
            f"Ensure the providing module is registered before the consuming module."
        )

    result = handler(**kwargs)

    # If the handler returned a coroutine, await it
    if hasattr(result, "__await__"):
        result = await result

    return result
