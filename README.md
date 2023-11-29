

![bitstring](https://raw.githubusercontent.com/scott-griffiths/bitstring/main/doc/bitstring_logo_small.png "bitstring")

**bitstring** is a Python module to help make the creation and analysis of binary data as simple and efficient as possible.


It has been maintained since 2006 and now has many millions of downloads per year.



[![CI badge](https://github.com/scott-griffiths/bitstring/actions/workflows/.github/workflows/ci.yml/badge.svg)](https://github.com/scott-griffiths/bitstring/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/readthedocs/bitstring)](https://bitstring.readthedocs.io/en/latest/)
[![Codacy Badge](https://img.shields.io/codacy/grade/8869499b2eed44548fa1a5149dd451f4)](https://app.codacy.com/gh/scott-griffiths/bitstring/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![Downloads](https://img.shields.io/pypi/dm/bitstring?color=blue)](https://pypistats.org/packages/bitstring) &nbsp; &nbsp; 
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/scott-griffiths/bitstring/main?labpath=doc%2Fwalkthrough.ipynb)

News
----
**29th November 2023**: bitstring 4.1.4 released. Version 4.1 is a large update in terms of how much of the code has changed.

* New Array class for homogeneous data.
* Support for 8-bit floating point values.
* Speed increased with bitarray dependency.

See the [release notes](https://github.com/scott-griffiths/bitstring/blob/main/release_notes.txt) for details. Please let me know if you encounter any problems.


Overview
--------

* Efficiently store and manipulate binary data in idiomatic Python.
* Create bitstrings from hex, octal, binary, files, formatted strings, bytes, integers and floats of different endiannesses.
* Powerful binary packing and unpacking functions.
* Bit-level slicing, joining, searching, replacing and more.
* Create and manipulate arrays of fixed-length bitstrings.
* Read from and interpret bitstrings as streams of binary data.
* Rich API - chances are that whatever you want to do there's a simple and elegant way of doing it.
* Open source software, released under the MIT licence.

Documentation
-------------
Extensive documentation for the bitstring module is available.
Some starting points are given below:

* [Overview](https://bitstring.readthedocs.io/en/stable/index.html)
* [Quick Reference](https://bitstring.readthedocs.io/en/stable/quick_reference.html)
* [Full Reference](https://bitstring.readthedocs.io/en/stable/reference.html)

You can also try out the interactive walkthrough notebook on [binder](https://mybinder.org/v2/gh/scott-griffiths/bitstring/main?labpath=doc%2Fwalkthrough.ipynb).

Release Notes
-------------

To see what been added, improved or fixed, and also to see what's coming in the next version, see the [release notes](https://github.com/scott-griffiths/bitstring/blob/main/release_notes.txt).

Examples
--------

### Installation

    $ pip install bitstring

### Creation

     >>> from bitstring import Bits, BitArray, BitStream, pack
     >>> a = BitArray(bin='00101')
     >>> b = Bits(a_file_object)
     >>> c = BitArray('0xff, 0b101, 0o65, uint6=22')
     >>> d = pack('intle16, hex=a, 0b1', 100, a='0x34f')
     >>> e = pack('<16h', *range(16))

### Different interpretations, slicing and concatenation

     >>> a = BitArray('0x3348')
     >>> a.hex, a.bin, a.uint, a.float, a.bytes
     ('3348', '0011001101001000', 13128, 0.2275390625, b'3H')
     >>> a[10:3:-1].bin
     '0101100'
     >>> '0b100' + 3*a
     BitArray('0x866906690669, 0b000')

### Reading data sequentially

     >>> b = BitStream('0x160120f')
     >>> b.read(12).hex
     '160'
     >>> b.pos = 0
     >>> b.read('uint12')
     352
     >>> b.readlist('uint12, bin3')
     [288, '111']

### Searching, inserting and deleting

     >>> c = BitArray('0b00010010010010001111')   # c.hex == '0x1248f'
     >>> c.find('0x48')
     (8,)
     >>> c.replace('0b001', '0xabc')
     >>> c.insert('0b0000', pos=3)
     >>> del c[12:16]

### Arrays of fixed-length formats

     >>> from bitstring import Array
     >>> a = Array('uint7', [9, 100, 3, 1])
     >>> a.data
     BitArray('0x1390181')
     >>> a[::2] *= 5
     >>> a
     Array('uint7', [45, 100, 15, 1])


Unit Tests
----------

The 700+ unit tests should all pass. They can be run from the root of the project with

     python -m unittest


Credits
-------

Created in 2006 to help with ad hoc parsing and creation of compressed video files.
Maintained and expanded ever since as it became unexpectedly popular.
Thanks to all those who have contributed ideas and code (and bug reports) over the years.


<sub>Copyright (c) 2006 - 2023 Scott Griffiths</sub>
