import array
import math
import struct
import bitarray
from bitstring.luts import mxfp_luts_compressed
import zlib
from typing import Optional

class MXFPFormat:
    """Defining an MXFP micro-scaling floating point format"""

    def __init__(self, exp_bits: int, mantissa_bits: int, bias: int, mxfp_overflow: str):
        self.exp_bits = exp_bits
        self.mantissa_bits = mantissa_bits
        self.bias = bias
        self.mxfp_overflow = mxfp_overflow
        self.pos_clamp_value = (1 << self.exp_bits + self.mantissa_bits) - 1
        self.neg_clamp_value = (1 << 1 + self.exp_bits + self.mantissa_bits) - 1
        if self.exp_bits == 4 and self.mantissa_bits == 3:
            if self.mxfp_overflow == 'saturate':
                self.pos_clamp_value = 126
                self.neg_clamp_value = 254
            else:
                self.pos_clamp_value = self.neg_clamp_value = 255
        if self.exp_bits == 5 and self.mantissa_bits == 2:
            if self.mxfp_overflow == 'saturate':
                self.pos_clamp_value = 123
                self.neg_clamp_value = 251
            else:
                self.pos_clamp_value = 124
                self.neg_clamp_value = 252
        self.lut_float16_to_mxfp = None
        self.lut_int_to_float = None

    def __str__(self):
        return f"MXFPFormat(exp_bits={self.exp_bits}, mantissa_bits={self.mantissa_bits}, bias={self.bias}, mxfp_overflow='{self.mxfp_overflow}')"

    def float_to_int(self, f: float) -> int:
        """Given a Python float convert to the best mxfp float (expressed as an int) that represents it."""
        pass

    def createLUT_for_int_to_float(self) -> array.array:
        """Create a LUT to convert an int in representing a MXFP float into a Python float"""
        pass

    def createLUT_for_float16_to_mxfp(self) -> bytes:
        """Create a LUT to convert a float16 into a MXFP format"""
        pass
e2m1mxfp_fmt = MXFPFormat(exp_bits=2, mantissa_bits=1, bias=1, mxfp_overflow='saturate')
e2m3mxfp_fmt = MXFPFormat(exp_bits=2, mantissa_bits=3, bias=1, mxfp_overflow='saturate')
e3m2mxfp_fmt = MXFPFormat(exp_bits=3, mantissa_bits=2, bias=3, mxfp_overflow='saturate')
e4m3mxfp_saturate_fmt = MXFPFormat(exp_bits=4, mantissa_bits=3, bias=7, mxfp_overflow='saturate')
e5m2mxfp_saturate_fmt = MXFPFormat(exp_bits=5, mantissa_bits=2, bias=15, mxfp_overflow='saturate')
e4m3mxfp_overflow_fmt = MXFPFormat(exp_bits=4, mantissa_bits=3, bias=7, mxfp_overflow='overflow')
e5m2mxfp_overflow_fmt = MXFPFormat(exp_bits=5, mantissa_bits=2, bias=15, mxfp_overflow='overflow')