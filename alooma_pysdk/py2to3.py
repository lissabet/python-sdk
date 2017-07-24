import six

try:
    import queue
except ImportError:
    import Queue as queue

if six.PY2:
    EMPTY_STRING = ""
else:
    EMPTY_STRING = b''


if six.PY2:
    from urllib2 import urlopen
    from urllib import urlencode
    from urlparse import urlparse
    from urllib2 import HTTPError
    str = str
    unicode = unicode
    long = long
    bytes = str
    basestring = (basestring, )
    reload = reload
else:
    str = str
    unicode = str
    bytes = bytes
    basestring = (str, bytes)
    long = int
    from urllib.request import urlopen
    from urllib.parse import urlencode
    from urllib.parse import urlparse
    from urllib.error import HTTPError
    from importlib import reload
