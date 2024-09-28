from __future__ import annotations
import functools
import re
from typing import Tuple, List, Optional, Pattern, Dict, Union, Match
NAME_INT_RE: Pattern[str] = re.compile('^([a-zA-Z][a-zA-Z0-9_]*?):?(\\d*)$')
NAME_KWARG_RE: Pattern[str] = re.compile('^([a-zA-Z][a-zA-Z0-9_]*?):?([a-zA-Z0-9_]+)$')
CACHE_SIZE = 256
DEFAULT_BITS: Pattern[str] = re.compile('^(?P<len>[^=]+)?(=(?P<value>.*))?$', re.IGNORECASE)
MULTIPLICATIVE_RE: Pattern[str] = re.compile('^(?P<factor>.*)\\*(?P<token>.+)')
LITERAL_RE: Pattern[str] = re.compile('^(?P<name>0([xob]))(?P<value>.+)', re.IGNORECASE)
STRUCT_PACK_RE: Pattern[str] = re.compile('^(?P<endian>[<>@=])(?P<fmt>(?:\\d*[bBhHlLqQefd])+)$')
BYTESWAP_STRUCT_PACK_RE: Pattern[str] = re.compile('^(?P<endian>[<>@=])?(?P<fmt>(?:\\d*[bBhHlLqQefd])+)$')
SINGLE_STRUCT_PACK_RE: Pattern[str] = re.compile('^(?P<endian>[<>@=])(?P<fmt>[bBhHlLqQefd])$')
STRUCT_SPLIT_RE: Pattern[str] = re.compile('\\d*[bBhHlLqQefd]')
REPLACEMENTS_BE: Dict[str, str] = {'b': 'int8', 'B': 'uint8', 'h': 'intbe16', 'H': 'uintbe16', 'l': 'intbe32', 'L': 'uintbe32', 'q': 'intbe64', 'Q': 'uintbe64', 'e': 'floatbe16', 'f': 'floatbe32', 'd': 'floatbe64'}
REPLACEMENTS_LE: Dict[str, str] = {'b': 'int8', 'B': 'uint8', 'h': 'intle16', 'H': 'uintle16', 'l': 'intle32', 'L': 'uintle32', 'q': 'intle64', 'Q': 'uintle64', 'e': 'floatle16', 'f': 'floatle32', 'd': 'floatle64'}
REPLACEMENTS_NE: Dict[str, str] = {'b': 'int8', 'B': 'uint8', 'h': 'intne16', 'H': 'uintne16', 'l': 'intne32', 'L': 'uintne32', 'q': 'intne64', 'Q': 'uintne64', 'e': 'floatne16', 'f': 'floatne32', 'd': 'floatne64'}
PACK_CODE_SIZE: Dict[str, int] = {'b': 1, 'B': 1, 'h': 2, 'H': 2, 'l': 4, 'L': 4, 'q': 8, 'Q': 8, 'e': 2, 'f': 4, 'd': 8}

def structparser(m: Match[str]) -> List[str]:
    """Parse struct-like format string token into sub-token list."""
    pass

@functools.lru_cache(CACHE_SIZE)
def tokenparser(fmt: str, keys: Tuple[str, ...]=()) -> Tuple[bool, List[Tuple[str, Union[int, str, None], Optional[str]]]]:
    """Divide the format string into tokens and parse them.

    Return stretchy token and list of [initialiser, length, value]
    initialiser is one of: hex, oct, bin, uint, int, se, ue, 0x, 0o, 0b etc.
    length is None if not known, as is value.

    If the token is in the keyword dictionary (keys) then it counts as a
    special case and isn't messed with.

    tokens must be of the form: [factor*][initialiser][:][length][=value]

    """
    pass
BRACKET_RE = re.compile('(?P<factor>\\d+)\\*\\(')

def expand_brackets(s: str) -> str:
    """Expand all brackets."""
    pass