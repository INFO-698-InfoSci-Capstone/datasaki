import json
import os as _os

from app.utils import read_file

_basepath = _os.path.dirname(__file__)
_languages = {}
for filename in _os.listdir(_basepath):
    lang, ext = _os.path.splitext(filename)
    if ext == ".json":
        filepath = _os.path.abspath(_os.path.join(_basepath, filename))
        _languages[lang] = json.loads(read_file(filepath))


def text(key):
    curr_lang = "en"
    return _languages.get(curr_lang, {}).get(key) or key
