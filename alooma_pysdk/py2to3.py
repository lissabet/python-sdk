import six

try:
    import queue
except ImportError:
    import Queue as queue

if six.PY2:
    str = str
    unicode = unicode
    long = long
    bytes = str
    basestring = (basestring, )
else:
    str = str
    unicode = str
    bytes = bytes
    basestring = (str, bytes)
    long = int
