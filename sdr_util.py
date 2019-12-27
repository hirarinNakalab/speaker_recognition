import toolz as tz
from toolz import curried as c

from htm.bindings.sdr import SDR


@tz.curry
def get_encoding(sdr, shape):
    encoding = SDR(dimensions=tuple(shape))
    encoding.dense = sdr
    return encoding

@tz.curry
def get_dense_array(val, enc, width):
    return tz.pipe(val,
                   c.map(lambda x: x*100),
                   c.map(enc.encode),
                   c.map(lambda x: getattr(x, 'dense')),
                   list,
                   get_encoding(shape=(val.shape[-1], width)))