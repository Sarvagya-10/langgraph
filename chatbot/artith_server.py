# arith_server.py

from __future__ import annotations

from fastmcp import FastMCP


# --------------------------------------------------
# MCP Server Instance
# --------------------------------------------------

mcp = FastMCP("arith")


# --------------------------------------------------
# Utility
# --------------------------------------------------

def _as_number(x) -> float:
    """
    Convert int/float or numeric string to float.
    Raise clear error otherwise.
    """
    if isinstance(x, (int, float)):
        return float(x)

    if isinstance(x, str):
        try:
            return float(x.strip())
        except ValueError:
            raise TypeError(f"Invalid numeric string: {x}")

    raise TypeError("Expected a number (int/float or numeric string)")


# --------------------------------------------------
# Tools
# --------------------------------------------------

@mcp.tool()
async def add(a: float, b: float) -> float:
    """Return a + b."""
    return _as_number(a) + _as_number(b)


@mcp.tool()
async def subtract(a: float, b: float) -> float:
    """Return a - b."""
    return _as_number(a) - _as_number(b)


@mcp.tool()
async def multiply(a: float, b: float) -> float:
    """Return a * b."""
    return _as_number(a) * _as_number(b)


@mcp.tool()
async def divide(a: float, b: float) -> float:
    """Return a / b."""
    denominator = _as_number(b)
    if denominator == 0:
        raise ValueError("Division by zero is not allowed")
    return _as_number(a) / denominator


# --------------------------------------------------
# Entrypoint
# --------------------------------------------------

if __name__ == "__main__":
    mcp.run()