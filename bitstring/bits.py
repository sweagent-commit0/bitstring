from __future__ import annotations
import numbers
import pathlib
import sys
import mmap
import struct
import array
import io
from collections import abc
import functools
from typing import Tuple, Union, List, Iterable, Any, Optional, BinaryIO, TextIO, overload, Iterator, Type, TypeVar
import bitarray
import bitarray.util
import bitstring
from bitstring.bitstore import BitStore
from bitstring import bitstore_helpers, utils
from bitstring.dtypes import Dtype, dtype_register
from bitstring.fp8 import p4binary_fmt, p3binary_fmt
from bitstring.mxfp import e3m2mxfp_fmt, e2m3mxfp_fmt, e2m1mxfp_fmt, e4m3mxfp_saturate_fmt, e5m2mxfp_saturate_fmt
from bitstring.bitstring_options import Colour
BitsType = Union['Bits', str, Iterable[Any], bool, BinaryIO, bytearray, bytes, memoryview, bitarray.bitarray]
TBits = TypeVar('TBits', bound='Bits')
MAX_CHARS: int = 250

class Bits:
    """A container holding an immutable sequence of bits.

    For a mutable container use the BitArray class instead.

    Methods:

    all() -- Check if all specified bits are set to 1 or 0.
    any() -- Check if any of specified bits are set to 1 or 0.
    copy() - Return a copy of the bitstring.
    count() -- Count the number of bits set to 1 or 0.
    cut() -- Create generator of constant sized chunks.
    endswith() -- Return whether the bitstring ends with a sub-string.
    find() -- Find a sub-bitstring in the current bitstring.
    findall() -- Find all occurrences of a sub-bitstring in the current bitstring.
    fromstring() -- Create a bitstring from a formatted string.
    join() -- Join bitstrings together using current bitstring.
    pp() -- Pretty print the bitstring.
    rfind() -- Seek backwards to find a sub-bitstring.
    split() -- Create generator of chunks split by a delimiter.
    startswith() -- Return whether the bitstring starts with a sub-bitstring.
    tobitarray() -- Return bitstring as a bitarray from the bitarray package.
    tobytes() -- Return bitstring as bytes, padding if needed.
    tofile() -- Write bitstring to file, padding if needed.
    unpack() -- Interpret bits using format string.

    Special methods:

    Also available are the operators [], ==, !=, +, *, ~, <<, >>, &, |, ^.

    Properties:

    [GENERATED_PROPERTY_DESCRIPTIONS]

    len -- Length of the bitstring in bits.

    """
    __slots__ = ('_bitstore', '_filename')

    def __init__(self, auto: Optional[Union[BitsType, int]]=None, /, length: Optional[int]=None, offset: Optional[int]=None, **kwargs) -> None:
        """Either specify an 'auto' initialiser:
        A string of comma separated tokens, an integer, a file object,
        a bytearray, a boolean iterable, an array or another bitstring.

        Or initialise via **kwargs with one (and only one) of:
        bin -- binary string representation, e.g. '0b001010'.
        hex -- hexadecimal string representation, e.g. '0x2ef'
        oct -- octal string representation, e.g. '0o777'.
        bytes -- raw data as a bytes object, for example read from a binary file.
        int -- a signed integer.
        uint -- an unsigned integer.
        float / floatbe -- a big-endian floating point number.
        bool -- a boolean (True or False).
        se -- a signed exponential-Golomb code.
        ue -- an unsigned exponential-Golomb code.
        sie -- a signed interleaved exponential-Golomb code.
        uie -- an unsigned interleaved exponential-Golomb code.
        floatle -- a little-endian floating point number.
        floatne -- a native-endian floating point number.
        bfloat / bfloatbe - a big-endian bfloat format 16-bit floating point number.
        bfloatle -- a little-endian bfloat format 16-bit floating point number.
        bfloatne -- a native-endian bfloat format 16-bit floating point number.
        intbe -- a signed big-endian whole byte integer.
        intle -- a signed little-endian whole byte integer.
        intne -- a signed native-endian whole byte integer.
        uintbe -- an unsigned big-endian whole byte integer.
        uintle -- an unsigned little-endian whole byte integer.
        uintne -- an unsigned native-endian whole byte integer.
        filename -- the path of a file which will be opened in binary read-only mode.

        Other keyword arguments:
        length -- length of the bitstring in bits, if needed and appropriate.
                  It must be supplied for all integer and float initialisers.
        offset -- bit offset to the data. These offset bits are
                  ignored and this is mainly intended for use when
                  initialising using 'bytes' or 'filename'.

        """
        self._bitstore.immutable = True

    def __new__(cls: Type[TBits], auto: Optional[Union[BitsType, int]]=None, /, length: Optional[int]=None, offset: Optional[int]=None, pos: Optional[int]=None, **kwargs) -> TBits:
        x = super().__new__(cls)
        if auto is None and (not kwargs):
            if length is not None:
                x._bitstore = BitStore(length)
                x._bitstore.setall(0)
            else:
                x._bitstore = BitStore()
            return x
        x._initialise(auto, length, offset, **kwargs)
        return x

    def __getattr__(self, attribute: str) -> Any:
        try:
            d = Dtype(attribute)
        except ValueError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attribute}'.")
        if d.bitlength is not None and len(self) != d.bitlength:
            raise ValueError(f"bitstring length {len(self)} doesn't match length {d.bitlength} of property '{attribute}'.")
        return d.get_fn(self)

    def __iter__(self) -> Iterable[bool]:
        return iter(self._bitstore)

    def __copy__(self: TBits) -> TBits:
        """Return a new copy of the Bits for the copy module."""
        return self

    def __lt__(self, other: Any) -> bool:
        return NotImplemented

    def __gt__(self, other: Any) -> bool:
        return NotImplemented

    def __le__(self, other: Any) -> bool:
        return NotImplemented

    def __ge__(self, other: Any) -> bool:
        return NotImplemented

    def __add__(self: TBits, bs: BitsType) -> TBits:
        """Concatenate bitstrings and return new bitstring.

        bs -- the bitstring to append.

        """
        bs = self.__class__._create_from_bitstype(bs)
        s = self._copy() if len(bs) <= len(self) else bs._copy()
        if len(bs) <= len(self):
            s._addright(bs)
        else:
            s._addleft(self)
        return s

    def __radd__(self: TBits, bs: BitsType) -> TBits:
        """Append current bitstring to bs and return new bitstring.

        bs -- An object that can be 'auto' initialised as a bitstring that will be appended to.

        """
        bs = self.__class__._create_from_bitstype(bs)
        return bs.__add__(self)

    @overload
    def __getitem__(self: TBits, key: slice, /) -> TBits:
        ...

    @overload
    def __getitem__(self, key: int, /) -> bool:
        ...

    def __getitem__(self: TBits, key: Union[slice, int], /) -> Union[TBits, bool]:
        """Return a new bitstring representing a slice of the current bitstring.

        Indices are in units of the step parameter (default 1 bit).
        Stepping is used to specify the number of bits in each item.

        >>> print(BitArray('0b00110')[1:4])
        '0b011'
        >>> print(BitArray('0x00112233')[1:3:8])
        '0x1122'

        """
        if isinstance(key, numbers.Integral):
            return bool(self._bitstore.getindex(key))
        bs = super().__new__(self.__class__)
        bs._bitstore = self._bitstore.getslice_withstep(key)
        return bs

    def __len__(self) -> int:
        """Return the length of the bitstring in bits."""
        return self._getlength()

    def __bytes__(self) -> bytes:
        return self.tobytes()

    def __str__(self) -> str:
        """Return approximate string representation of bitstring for printing.

        Short strings will be given wholly in hexadecimal or binary. Longer
        strings may be part hexadecimal and part binary. Very long strings will
        be truncated with '...'.

        """
        length = len(self)
        if not length:
            return ''
        if length > MAX_CHARS * 4:
            return ''.join(('0x', self[0:MAX_CHARS * 4]._gethex(), '...'))
        if length < 32 and length % 4 != 0:
            return '0b' + self.bin
        if not length % 4:
            return '0x' + self.hex
        bits_at_end = length % 4
        return ''.join(('0x', self[0:length - bits_at_end]._gethex(), ', ', '0b', self[length - bits_at_end:]._getbin()))

    def __repr__(self) -> str:
        """Return representation that could be used to recreate the bitstring.

        If the returned string is too long it will be truncated. See __str__().

        """
        return self._repr(self.__class__.__name__, len(self), 0)

    def __eq__(self, bs: Any, /) -> bool:
        """Return True if two bitstrings have the same binary representation.

        >>> BitArray('0b1110') == '0xe'
        True

        """
        try:
            return self._bitstore == Bits._create_from_bitstype(bs)._bitstore
        except TypeError:
            return False

    def __ne__(self, bs: Any, /) -> bool:
        """Return False if two bitstrings have the same binary representation.

        >>> BitArray('0b111') == '0x7'
        False

        """
        return not self.__eq__(bs)

    def __invert__(self: TBits) -> TBits:
        """Return bitstring with every bit inverted.

        Raises Error if the bitstring is empty.

        """
        if len(self) == 0:
            raise bitstring.Error('Cannot invert empty bitstring.')
        s = self._copy()
        s._invert_all()
        return s

    def __lshift__(self: TBits, n: int, /) -> TBits:
        """Return bitstring with bits shifted by n to the left.

        n -- the number of bits to shift. Must be >= 0.

        """
        if n < 0:
            raise ValueError('Cannot shift by a negative amount.')
        if len(self) == 0:
            raise ValueError('Cannot shift an empty bitstring.')
        n = min(n, len(self))
        s = self._absolute_slice(n, len(self))
        s._addright(Bits(n))
        return s

    def __rshift__(self: TBits, n: int, /) -> TBits:
        """Return bitstring with bits shifted by n to the right.

        n -- the number of bits to shift. Must be >= 0.

        """
        if n < 0:
            raise ValueError('Cannot shift by a negative amount.')
        if len(self) == 0:
            raise ValueError('Cannot shift an empty bitstring.')
        if not n:
            return self._copy()
        s = self.__class__(length=min(n, len(self)))
        n = min(n, len(self))
        s._addright(self._absolute_slice(0, len(self) - n))
        return s

    def __mul__(self: TBits, n: int, /) -> TBits:
        """Return bitstring consisting of n concatenations of self.

        Called for expression of the form 'a = b*3'.
        n -- The number of concatenations. Must be >= 0.

        """
        if n < 0:
            raise ValueError('Cannot multiply by a negative integer.')
        if not n:
            return self.__class__()
        s = self._copy()
        s._imul(n)
        return s

    def __rmul__(self: TBits, n: int, /) -> TBits:
        """Return bitstring consisting of n concatenations of self.

        Called for expressions of the form 'a = 3*b'.
        n -- The number of concatenations. Must be >= 0.

        """
        return self.__mul__(n)

    def __and__(self: TBits, bs: BitsType, /) -> TBits:
        """Bit-wise 'and' between two bitstrings. Returns new bitstring.

        bs -- The bitstring to '&' with.

        Raises ValueError if the two bitstrings have differing lengths.

        """
        if bs is self:
            return self.copy()
        bs = Bits._create_from_bitstype(bs)
        s = object.__new__(self.__class__)
        s._bitstore = self._bitstore & bs._bitstore
        return s

    def __rand__(self: TBits, bs: BitsType, /) -> TBits:
        """Bit-wise 'and' between two bitstrings. Returns new bitstring.

        bs -- the bitstring to '&' with.

        Raises ValueError if the two bitstrings have differing lengths.

        """
        return self.__and__(bs)

    def __or__(self: TBits, bs: BitsType, /) -> TBits:
        """Bit-wise 'or' between two bitstrings. Returns new bitstring.

        bs -- The bitstring to '|' with.

        Raises ValueError if the two bitstrings have differing lengths.

        """
        if bs is self:
            return self.copy()
        bs = Bits._create_from_bitstype(bs)
        s = object.__new__(self.__class__)
        s._bitstore = self._bitstore | bs._bitstore
        return s

    def __ror__(self: TBits, bs: BitsType, /) -> TBits:
        """Bit-wise 'or' between two bitstrings. Returns new bitstring.

        bs -- The bitstring to '|' with.

        Raises ValueError if the two bitstrings have differing lengths.

        """
        return self.__or__(bs)

    def __xor__(self: TBits, bs: BitsType, /) -> TBits:
        """Bit-wise 'xor' between two bitstrings. Returns new bitstring.

        bs -- The bitstring to '^' with.

        Raises ValueError if the two bitstrings have differing lengths.

        """
        bs = Bits._create_from_bitstype(bs)
        s = object.__new__(self.__class__)
        s._bitstore = self._bitstore ^ bs._bitstore
        return s

    def __rxor__(self: TBits, bs: BitsType, /) -> TBits:
        """Bit-wise 'xor' between two bitstrings. Returns new bitstring.

        bs -- The bitstring to '^' with.

        Raises ValueError if the two bitstrings have differing lengths.

        """
        return self.__xor__(bs)

    def __contains__(self, bs: BitsType, /) -> bool:
        """Return whether bs is contained in the current bitstring.

        bs -- The bitstring to search for.

        """
        found = Bits.find(self, bs, bytealigned=False)
        return bool(found)

    def __hash__(self) -> int:
        """Return an integer hash of the object."""
        if len(self) <= 2000:
            return hash((self.tobytes(), len(self)))
        else:
            return hash(((self[:800] + self[-800:]).tobytes(), len(self)))

    def __bool__(self) -> bool:
        """Return False if bitstring is empty, otherwise return True."""
        return len(self) != 0

    def _clear(self) -> None:
        """Reset the bitstring to an empty state."""
        self._bitstore = BitStore()

    def _setauto_no_length_or_offset(self, s: BitsType, /) -> None:
        """Set bitstring from a bitstring, file, bool, array, iterable or string."""
        if isinstance(s, Bits):
            self._bitstore = s._bitstore.copy()
        elif isinstance(s, (bytes, bytearray)):
            self._setbytes(s)
        elif isinstance(s, str):
            self._setbin(s)
        elif isinstance(s, (int, float)):
            self._setuint(int(s))
        elif hasattr(s, 'read'):  # file-like object
            self._setfile(s)
        elif isinstance(s, (list, tuple)):
            self._setbin(''.join(str(int(bool(b))) for b in s))
        else:
            raise TypeError(f"Cannot create bitstring from {type(s)} type.")

    def _setauto(self, s: BitsType, length: Optional[int], offset: Optional[int], /) -> None:
        """Set bitstring from a bitstring, file, bool, array, iterable or string."""
        if isinstance(s, Bits):
            if length is None:
                length = len(s) - (offset or 0)
            self._bitstore = s._bitstore[offset:offset+length] if offset else s._bitstore[:length]
        elif isinstance(s, (bytes, bytearray)):
            self._setbytes_with_truncation(s, length, offset)
        elif isinstance(s, str):
            if offset is not None or length is not None:
                s = s[offset:offset+length] if offset else s[:length]
            self._setbin(s)
        elif isinstance(s, (int, float)):
            if offset is not None:
                raise TypeError("Cannot use offset with integer initialisation.")
            self._setuint(int(s), length)
        elif hasattr(s, 'read'):  # file-like object
            self._setfile(s, length, offset)
        elif isinstance(s, (list, tuple)):
            if offset is not None or length is not None:
                s = s[offset:offset+length] if offset else s[:length]
            self._setbin(''.join(str(int(bool(b))) for b in s))
        else:
            raise TypeError(f"Cannot create bitstring from {type(s)} type.")

    def _setfile(self, filename: str, length: Optional[int]=None, offset: Optional[int]=None) -> None:
        """Use file as source of bits."""
        with open(filename, 'rb') as f:
            if offset is not None:
                f.seek(offset // 8)
            data = f.read(length // 8 if length is not None else -1)
        self._setbytes(data)
        if offset is not None and offset % 8:
            self._bitstore = self._bitstore[offset % 8:]
        if length is not None:
            self._bitstore = self._bitstore[:length]

    def _setbytes(self, data: Union[bytearray, bytes, List], length: None=None) -> None:
        """Set the data from a bytes or bytearray object."""
        if isinstance(data, list):
            data = bytes(data)
        if length is None:
            length = len(data) * 8
        self._bitstore = BitStore(data, length)

    def _setbytes_with_truncation(self, data: Union[bytearray, bytes], length: Optional[int]=None, offset: Optional[int]=None) -> None:
        """Set the data from a bytes or bytearray object, with optional offset and length truncations."""
        if offset is not None:
            data = data[offset // 8:]
            offset = offset % 8
        else:
            offset = 0
        
        if length is not None:
            data = data[:((length + offset + 7) // 8)]
        
        self._setbytes(data)
        
        if offset:
            self._bitstore = self._bitstore[offset:]
        
        if length is not None:
            self._bitstore = self._bitstore[:length]

    def _getbytes(self) -> bytes:
        """Return the data as an ordinary bytes object."""
        return self._bitstore.getbytes()
    _unprintable = list(range(0, 32))
    _unprintable.extend(range(127, 255))

    def _getbytes_printable(self) -> str:
        """Return an approximation of the data as a string of printable characters."""
        bytes_data = self._getbytes()
        return ''.join(chr(b) if b not in self._unprintable else '.' for b in bytes_data)

    def _setuint(self, uint: int, length: Optional[int]=None) -> None:
        """Reset the bitstring to have given unsigned int interpretation."""
        if length is None:
            length = uint.bit_length()
        elif uint >= (1 << length):
            raise ValueError(f"uint {uint} is too large for a bitstring of length {length}.")
        self._bitstore = BitStore.frombytes(uint.to_bytes((length + 7) // 8, byteorder='big'), length)

    def _getuint(self) -> int:
        """Return data as an unsigned int."""
        return int.from_bytes(self._getbytes(), byteorder='big')

    def _setint(self, int_: int, length: Optional[int]=None) -> None:
        """Reset the bitstring to have given signed int interpretation."""
        if length is None:
            length = int_.bit_length() + 1  # +1 for sign bit
        elif int_ >= (1 << (length - 1)) or int_ < -(1 << (length - 1)):
            raise ValueError(f"int {int_} is too large for a bitstring of length {length}.")
        # Convert to two's complement representation
        if int_ < 0:
            int_ = (1 << length) + int_
        self._bitstore = BitStore.frombytes(int_.to_bytes((length + 7) // 8, byteorder='big', signed=False), length)

    def _getint(self) -> int:
        """Return data as a two's complement signed int."""
        value = int.from_bytes(self._getbytes(), byteorder='big', signed=False)
        bits = len(self)
        if value & (1 << (bits - 1)):
            value -= 1 << bits
        return value

    def _setuintbe(self, uintbe: int, length: Optional[int]=None) -> None:
        """Set the bitstring to a big-endian unsigned int interpretation."""
        self._setuint(uintbe, length)

    def _getuintbe(self) -> int:
        """Return data as a big-endian two's complement unsigned int."""
        return self._getuint()

    def _setintbe(self, intbe: int, length: Optional[int]=None) -> None:
        """Set bitstring to a big-endian signed int interpretation."""
        self._setint(intbe, length)

    def _getintbe(self) -> int:
        """Return data as a big-endian two's complement signed int."""
        return self._getint()

    def _getuintle(self) -> int:
        """Interpret as a little-endian unsigned int."""
        return int.from_bytes(self._getbytes(), byteorder='little')

    def _getintle(self) -> int:
        """Interpret as a little-endian signed int."""
        value = int.from_bytes(self._getbytes(), byteorder='little', signed=False)
        bits = len(self)
        if value & (1 << (bits - 1)):
            value -= 1 << bits
        return value

    def _getfloatbe(self) -> float:
        """Interpret the whole bitstring as a big-endian float."""
        import struct
        if len(self) == 32:
            return struct.unpack('>f', self._getbytes())[0]
        elif len(self) == 64:
            return struct.unpack('>d', self._getbytes())[0]
        else:
            raise ValueError("Bitstring length must be 32 or 64 bits for float interpretation")

    def _getfloatle(self) -> float:
        """Interpret the whole bitstring as a little-endian float."""
        import struct
        if len(self) == 32:
            return struct.unpack('<f', self._getbytes())[0]
        elif len(self) == 64:
            return struct.unpack('<d', self._getbytes())[0]
        else:
            raise ValueError("Bitstring length must be 32 or 64 bits for float interpretation")

    def _setue(self, i: int) -> None:
        """Initialise bitstring with unsigned exponential-Golomb code for integer i.

        Raises ValueError if i < 0.
        """
        if i < 0:
            raise ValueError("Cannot use negative initializer for unsigned exponential-Golomb.")
        
        length = i.bit_length()
        num_zeros = length - 1
        code = '0' * num_zeros + '1' + format(i, f'0{length}b')[1:]
        self._setbin(code)

    def _readue(self, pos: int) -> Tuple[int, int]:
        """Return interpretation of next bits as unsigned exponential-Golomb code.

        Raises ValueError if the end of the bitstring is encountered while
        reading the code.
        """
        try:
            # Read leading zeros
            leadingzeros = self._bitstore.find('0b1', pos) - pos
            pos += leadingzeros + 1
            
            if leadingzeros == 0:
                return 0, pos
            
            # Read the rest of the code
            value = self._readuint(leadingzeros, pos)
            pos += leadingzeros
            
            return (1 << leadingzeros) - 1 + value, pos
        except ValueError:
            raise ValueError("End of bitstring reached while reading exponential-Golomb code.")

    def _setse(self, i: int) -> None:
        """Initialise bitstring with signed exponential-Golomb code for integer i."""
        if i == 0:
            self._setue(0)
        else:
            coded_i = (abs(i) << 1) - 1 if i > 0 else abs(i) << 1
            self._setue(coded_i)

    def _readse(self, pos: int) -> Tuple[int, int]:
        """Return interpretation of next bits as a signed exponential-Golomb code.

        Advances position to after the read code.

        Raises ValueError if the end of the bitstring is encountered while
        reading the code.
        """
        try:
            value, newpos = self._readue(pos)
            if value & 1:
                return (value + 1) >> 1, newpos
            else:
                return -(value >> 1), newpos
        except ValueError:
            raise ValueError("End of bitstring reached while reading signed exponential-Golomb code.")

    def _setuie(self, i: int) -> None:
        """Initialise bitstring with unsigned interleaved exponential-Golomb code for integer i.

        Raises ValueError if i < 0.
        """
        if i < 0:
            raise ValueError("Cannot use negative initializer for unsigned interleaved exponential-Golomb.")
        
        if i == 0:
            self._setbin('1')
            return
        
        binary = bin(i)[2:]  # Remove '0b' prefix
        length = len(binary)
        
        # Interleave zeros
        interleaved = ''.join('0' + bit for bit in binary[:-1]) + '1' + binary[-1]
        
        # Add leading zeros
        code = '0' * (length - 1) + interleaved
        
        self._setbin(code)

    def _readuie(self, pos: int) -> Tuple[int, int]:
        """Return interpretation of next bits as unsigned interleaved exponential-Golomb code.

        Raises ValueError if the end of the bitstring is encountered while
        reading the code.
        """
        try:
            # Read leading zeros
            leadingzeros = self._bitstore.find('0b1', pos) - pos
            pos += leadingzeros + 1
            
            if leadingzeros == 0:
                return 0, pos
            
            # Read interleaved bits
            value = 1
            for _ in range(leadingzeros):
                bit = self._readbool(pos)
                pos += 1
                value = (value << 1) | bit
            
            return value - 1, pos
        except ValueError:
            raise ValueError("End of bitstring reached while reading unsigned interleaved exponential-Golomb code.")

    def _setsie(self, i: int) -> None:
        """Initialise bitstring with signed interleaved exponential-Golomb code for integer i."""
        if i == 0:
            self._setuie(0)
        else:
            coded_i = (abs(i) << 1) - 1 if i > 0 else abs(i) << 1
            self._setuie(coded_i)

    def _readsie(self, pos: int) -> Tuple[int, int]:
        """Return interpretation of next bits as a signed interleaved exponential-Golomb code.

        Advances position to after the read code.

        Raises ValueError if the end of the bitstring is encountered while
        reading the code.
        """
        try:
            value, newpos = self._readuie(pos)
            if value & 1:
                return (value + 1) >> 1, newpos
            else:
                return -(value >> 1), newpos
        except ValueError:
            raise ValueError("End of bitstring reached while reading signed interleaved exponential-Golomb code.")

    def _setbin_safe(self, binstring: str, length: Optional[int]=None) -> None:
        """Reset the bitstring to the value given in binstring."""
        if not set(binstring).issubset('01'):
            raise ValueError("binstring can only contain '0' and '1'")
        if length is not None and len(binstring) > length:
            raise ValueError("binstring is too long for the specified length")
        
        self._bitstore = BitStore(bytes(int(binstring[i:i+8], 2) for i in range(0, len(binstring), 8)),
                                  length or len(binstring))

    def _setbin_unsafe(self, binstring: str, length: Optional[int]=None) -> None:
        """Same as _setbin_safe, but input isn't sanity checked. binstring mustn't start with '0b'."""
        self._bitstore = BitStore(bytes(int(binstring[i:i+8], 2) for i in range(0, len(binstring), 8)),
                                  length or len(binstring))

    def _getbin(self) -> str:
        """Return interpretation as a binary string."""
        return ''.join(format(byte, '08b') for byte in self._bitstore.getbytes())[:len(self)]

    def _setoct(self, octstring: str, length: Optional[int]=None) -> None:
        """Reset the bitstring to have the value given in octstring."""
        binstring = ''.join(format(int(c, 8), '03b') for c in octstring)
        self._setbin_unsafe(binstring, length)

    def _getoct(self) -> str:
        """Return interpretation as an octal string."""
        binary = self._getbin()
        # Pad the binary string to make its length a multiple of 3
        padded_binary = '0' * ((3 - len(binary) % 3) % 3) + binary
        return ''.join(str(int(padded_binary[i:i+3], 2)) for i in range(0, len(padded_binary), 3))

    def _sethex(self, hexstring: str, length: Optional[int]=None) -> None:
        """Reset the bitstring to have the value given in hexstring."""
        binstring = ''.join(format(int(c, 16), '04b') for c in hexstring)
        self._setbin_unsafe(binstring, length)

    def _gethex(self) -> str:
        """Return the hexadecimal representation as a string.

        Raises a ValueError if the bitstring's length is not a multiple of 4.
        """
        if len(self) % 4 != 0:
            raise ValueError("Cannot interpret as hex string - length is not a multiple of 4.")
        return ''.join(format(int(self._getbin()[i:i+4], 2), 'X') for i in range(0, len(self), 4))

    def _getlength(self) -> int:
        """Return the length of the bitstring in bits."""
        return len(self._bitstore)

    def _copy(self: TBits) -> TBits:
        """Create and return a new copy of the Bits (always in memory)."""
        new_bits = self.__class__()
        new_bits._bitstore = self._bitstore.copy()
        return new_bits

    def _slice(self: TBits, start: int, end: int) -> TBits:
        """Used internally to get a slice, without error checking."""
        new_bits = self.__class__()
        new_bits._bitstore = self._bitstore[start:end]
        return new_bits

    def _absolute_slice(self: TBits, start: int, end: int) -> TBits:
        """Used internally to get a slice, without error checking.
        Uses MSB0 bit numbering even if LSB0 is set."""
        return self._slice(start, end)

    def _readtoken(self, name: str, pos: int, length: Optional[int]) -> Tuple[Union[float, int, str, None, Bits], int]:
        """Reads a token from the bitstring and returns the result."""
        if name == 'uint':
            if length is None:
                raise ValueError("Token 'uint' requires a length")
            value = self._readuint(length, pos)
            return value, pos + length
        elif name == 'int':
            if length is None:
                raise ValueError("Token 'int' requires a length")
            value = self._readint(length, pos)
            return value, pos + length
        elif name == 'float':
            if length is None:
                raise ValueError("Token 'float' requires a length")
            value = self._readfloat(length, pos)
            return value, pos + length
        elif name == 'bytes':
            if length is None:
                raise ValueError("Token 'bytes' requires a length")
            value = self._readbytes(length, pos)
            return value, pos + length * 8
        elif name == 'bits':
            if length is None:
                raise ValueError("Token 'bits' requires a length")
            value = self._slice(pos, pos + length)
            return value, pos + length
        else:
            raise ValueError(f"Unknown token name: {name}")

    def _addright(self, bs: Bits, /) -> None:
        """Add a bitstring to the RHS of the current bitstring."""
        self._bitstore.append(bs._bitstore)

    def _addleft(self, bs: Bits, /) -> None:
        """Prepend a bitstring to the current bitstring."""
        new_bitstore = bs._bitstore.copy()
        new_bitstore.append(self._bitstore)
        self._bitstore = new_bitstore

    def _truncateleft(self: TBits, bits: int, /) -> TBits:
        """Truncate bits from the start of the bitstring. Return the truncated bits."""
        truncated = self._slice(0, bits)
        self._bitstore = self._bitstore[bits:]
        return truncated

    def _truncateright(self: TBits, bits: int, /) -> TBits:
        """Truncate bits from the end of the bitstring. Return the truncated bits."""
        length = len(self)
        truncated = self._slice(length - bits, length)
        self._bitstore = self._bitstore[:length - bits]
        return truncated

    def _insert(self, bs: Bits, pos: int, /) -> None:
        """Insert bs at pos."""
        left = self._slice(0, pos)
        right = self._slice(pos, len(self))
        self._bitstore = left._bitstore
        self._bitstore.append(bs._bitstore)
        self._bitstore.append(right._bitstore)

    def _overwrite(self, bs: Bits, pos: int, /) -> None:
        """Overwrite with bs at pos."""
        end = pos + len(bs)
        if end > len(self):
            raise ValueError("Cannot overwrite past the end of the bitstring")
        self._bitstore[pos:end] = bs._bitstore

    def _delete(self, bits: int, pos: int, /) -> None:
        """Delete bits at pos."""
        if pos + bits > len(self):
            raise ValueError("Cannot delete past the end of the bitstring")
        self._bitstore = self._bitstore[:pos] + self._bitstore[pos + bits:]

    def _reversebytes(self, start: int, end: int) -> None:
        """Reverse bytes in-place."""
        if start % 8 != 0 or end % 8 != 0:
            raise ValueError("Start and end positions must be byte-aligned")
        byte_start = start // 8
        byte_end = end // 8
        byte_data = bytearray(self._bitstore.getbytes())
        byte_data[byte_start:byte_end] = byte_data[byte_start:byte_end][::-1]
        self._bitstore = BitStore(bytes(byte_data), len(self))

    def _invert(self, pos: int, /) -> None:
        """Flip bit at pos 1<->0."""
        if pos < 0 or pos >= len(self):
            raise IndexError("Bit position out of range")
        byte_pos = pos // 8
        bit_pos = pos % 8
        byte_data = bytearray(self._bitstore.getbytes())
        byte_data[byte_pos] ^= (1 << (7 - bit_pos))
        self._bitstore = BitStore(bytes(byte_data), len(self))

    def _invert_all(self) -> None:
        """Invert every bit."""
        byte_data = bytearray(self._bitstore.getbytes())
        for i in range(len(byte_data)):
            byte_data[i] = ~byte_data[i] & 0xFF
        self._bitstore = BitStore(bytes(byte_data), len(self))

    def _ilshift(self: TBits, n: int, /) -> TBits:
        """Shift bits by n to the left in place. Return self."""
        if n == 0:
            return self
        if n < 0:
            return self._irshift(-n)
        if n >= len(self):
            self._clear()
            return self
        self._bitstore = self._bitstore[n:] + BitStore(b'\x00' * ((n + 7) // 8), n)
        return self

    def _irshift(self: TBits, n: int, /) -> TBits:
        """Shift bits by n to the right in place. Return self."""
        if n == 0:
            return self
        if n < 0:
            return self._ilshift(-n)
        if n >= len(self):
            self._clear()
            return self
        self._bitstore = BitStore(b'\x00' * ((n + 7) // 8), n) + self._bitstore[:-n]
        return self

    def _imul(self: TBits, n: int, /) -> TBits:
        """Concatenate n copies of self in place. Return self."""
        if n <= 0:
            self._clear()
        elif n > 1:
            original = self._bitstore.copy()
            for _ in range(n - 1):
                self._bitstore.append(original)
        return self

    def _validate_slice(self, start: Optional[int], end: Optional[int]) -> Tuple[int, int]:
        """Validate start and end and return them as positive bit positions."""
        pass

    def unpack(self, fmt: Union[str, List[Union[str, int]]], **kwargs) -> List[Union[int, float, str, Bits, bool, bytes, None]]:
        """Interpret the whole bitstring using fmt and return list.

        fmt -- A single string or a list of strings with comma separated tokens
               describing how to interpret the bits in the bitstring. Items
               can also be integers, for reading new bitstring of the given length.
        kwargs -- A dictionary or keyword-value pairs - the keywords used in the
                  format string will be replaced with their given value.

        Raises ValueError if the format is not understood. If not enough bits
        are available then all bits to the end of the bitstring will be used.

        See the docstring for 'read' for token examples.

        """
        pass

    def find(self, bs: BitsType, /, start: Optional[int]=None, end: Optional[int]=None, bytealigned: Optional[bool]=None) -> Union[Tuple[int], Tuple[()]]:
        """Find first occurrence of substring bs.

        Returns a single item tuple with the bit position if found, or an
        empty tuple if not found. The bit position (pos property) will
        also be set to the start of the substring if it is found.

        bs -- The bitstring to find.
        start -- The bit position to start the search. Defaults to 0.
        end -- The bit position one past the last bit to search.
               Defaults to len(self).
        bytealigned -- If True the bitstring will only be
                       found on byte boundaries.

        Raises ValueError if bs is empty, if start < 0, if end > len(self) or
        if end < start.

        >>> BitArray('0xc3e').find('0b1111')
        (6,)

        """
        pass

    def _find_msb0(self, bs: Bits, start: int, end: int, bytealigned: bool) -> Union[Tuple[int], Tuple[()]]:
        """Find first occurrence of a binary string."""
        pass

    def findall(self, bs: BitsType, start: Optional[int]=None, end: Optional[int]=None, count: Optional[int]=None, bytealigned: Optional[bool]=None) -> Iterable[int]:
        """Find all occurrences of bs. Return generator of bit positions.

        bs -- The bitstring to find.
        start -- The bit position to start the search. Defaults to 0.
        end -- The bit position one past the last bit to search.
               Defaults to len(self).
        count -- The maximum number of occurrences to find.
        bytealigned -- If True the bitstring will only be found on
                       byte boundaries.

        Raises ValueError if bs is empty, if start < 0, if end > len(self) or
        if end < start.

        Note that all occurrences of bs are found, even if they overlap.

        """
        pass

    def rfind(self, bs: BitsType, /, start: Optional[int]=None, end: Optional[int]=None, bytealigned: Optional[bool]=None) -> Union[Tuple[int], Tuple[()]]:
        """Find final occurrence of substring bs.

        Returns a single item tuple with the bit position if found, or an
        empty tuple if not found. The bit position (pos property) will
        also be set to the start of the substring if it is found.

        bs -- The bitstring to find.
        start -- The bit position to end the reverse search. Defaults to 0.
        end -- The bit position one past the first bit to reverse search.
               Defaults to len(self).
        bytealigned -- If True the bitstring will only be found on byte
                       boundaries.

        Raises ValueError if bs is empty, if start < 0, if end > len(self) or
        if end < start.

        """
        pass

    def _rfind_msb0(self, bs: Bits, start: int, end: int, bytealigned: bool) -> Union[Tuple[int], Tuple[()]]:
        """Find final occurrence of a binary string."""
        pass

    def cut(self, bits: int, start: Optional[int]=None, end: Optional[int]=None, count: Optional[int]=None) -> Iterator[Bits]:
        """Return bitstring generator by cutting into bits sized chunks.

        bits -- The size in bits of the bitstring chunks to generate.
        start -- The bit position to start the first cut. Defaults to 0.
        end -- The bit position one past the last bit to use in the cut.
               Defaults to len(self).
        count -- If specified then at most count items are generated.
                 Default is to cut as many times as possible.

        """
        pass

    def split(self, delimiter: BitsType, start: Optional[int]=None, end: Optional[int]=None, count: Optional[int]=None, bytealigned: Optional[bool]=None) -> Iterable[Bits]:
        """Return bitstring generator by splitting using a delimiter.

        The first item returned is the initial bitstring before the delimiter,
        which may be an empty bitstring.

        delimiter -- The bitstring used as the divider.
        start -- The bit position to start the split. Defaults to 0.
        end -- The bit position one past the last bit to use in the split.
               Defaults to len(self).
        count -- If specified then at most count items are generated.
                 Default is to split as many times as possible.
        bytealigned -- If True splits will only occur on byte boundaries.

        Raises ValueError if the delimiter is empty.

        """
        pass

    def join(self: TBits, sequence: Iterable[Any]) -> TBits:
        """Return concatenation of bitstrings joined by self.

        sequence -- A sequence of bitstrings.

        """
        pass

    def tobytes(self) -> bytes:
        """Return the bitstring as bytes, padding with zero bits if needed.

        Up to seven zero bits will be added at the end to byte align.

        """
        pass

    def tobitarray(self) -> bitarray.bitarray:
        """Convert the bitstring to a bitarray object."""
        pass

    def tofile(self, f: BinaryIO) -> None:
        """Write the bitstring to a file object, padding with zero bits if needed.

        Up to seven zero bits will be added at the end to byte align.

        """
        pass

    def startswith(self, prefix: BitsType, start: Optional[int]=None, end: Optional[int]=None) -> bool:
        """Return whether the current bitstring starts with prefix.

        prefix -- The bitstring to search for.
        start -- The bit position to start from. Defaults to 0.
        end -- The bit position to end at. Defaults to len(self).

        """
        pass

    def endswith(self, suffix: BitsType, start: Optional[int]=None, end: Optional[int]=None) -> bool:
        """Return whether the current bitstring ends with suffix.

        suffix -- The bitstring to search for.
        start -- The bit position to start from. Defaults to 0.
        end -- The bit position to end at. Defaults to len(self).

        """
        pass

    def all(self, value: Any, pos: Optional[Iterable[int]]=None) -> bool:
        """Return True if one or many bits are all set to bool(value).

        value -- If value is True then checks for bits set to 1, otherwise
                 checks for bits set to 0.
        pos -- An iterable of bit positions. Negative numbers are treated in
               the same way as slice indices. Defaults to the whole bitstring.

        """
        pass

    def any(self, value: Any, pos: Optional[Iterable[int]]=None) -> bool:
        """Return True if any of one or many bits are set to bool(value).

        value -- If value is True then checks for bits set to 1, otherwise
                 checks for bits set to 0.
        pos -- An iterable of bit positions. Negative numbers are treated in
               the same way as slice indices. Defaults to the whole bitstring.

        """
        pass

    def count(self, value: Any) -> int:
        """Return count of total number of either zero or one bits.

        value -- If bool(value) is True then bits set to 1 are counted, otherwise bits set
                 to 0 are counted.

        >>> Bits('0xef').count(1)
        7

        """
        pass

    @staticmethod
    def _chars_per_group(bits_per_group: int, fmt: Optional[str]):
        """How many characters are needed to represent a number of bits with a given format."""
        pass

    @staticmethod
    def _bits_per_char(fmt: str):
        """How many bits are represented by each character of a given format."""
        pass

    def _pp(self, dtype1: Dtype, dtype2: Optional[Dtype], bits_per_group: int, width: int, sep: str, format_sep: str, show_offset: bool, stream: TextIO, lsb0: bool, offset_factor: int) -> None:
        """Internal pretty print method."""
        pass

    def pp(self, fmt: Optional[str]=None, width: int=120, sep: str=' ', show_offset: bool=True, stream: TextIO=sys.stdout) -> None:
        """Pretty print the bitstring's value.

        fmt -- Printed data format. One or two of 'bin', 'oct', 'hex' or 'bytes'.
              The number of bits represented in each printed group defaults to 8 for hex and bin,
              12 for oct and 32 for bytes. This can be overridden with an explicit length, e.g. 'hex:64'.
              Use a length of 0 to not split into groups, e.g. `bin:0`.
        width -- Max width of printed lines. Defaults to 120. A single group will always be printed
                 per line even if it exceeds the max width.
        sep -- A separator string to insert between groups. Defaults to a single space.
        show_offset -- If True (the default) shows the bit offset in the first column of each line.
        stream -- A TextIO object with a write() method. Defaults to sys.stdout.

        >>> s.pp('hex16')
        >>> s.pp('b, h', sep='_', show_offset=False)

        """
        pass

    def copy(self: TBits) -> TBits:
        """Return a copy of the bitstring."""
        pass

    @classmethod
    def fromstring(cls: TBits, s: str, /) -> TBits:
        """Create a new bitstring from a formatted string."""
        pass
    len = length = property(_getlength, doc='The length of the bitstring in bits. Read only.')
