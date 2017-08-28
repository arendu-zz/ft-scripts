#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Rules governing the use of punctuation in scripts.
"""

import re

# Punctuation characters that do *not* appear as part of a valid word in the
# language. The app uses this to determine:
#
#  1. What characters should be ignored (allowing the user to input random
#     punctuation)
#  2. What characters should auto-complete the word the user is currently
#     typing.
#
# XXX Make this language-dependent as soon as we support more than French.
IGNORED_PUNCTUATION_FRENCH = unicode("[]{}<>)(^~_‐;!¡|?¿/·\"«»“”„:,.▶*+。""，、､‚｡；：「」『』（）"
        "？！…《》〈〉．～،؟।‧", "utf-8")

PUNCTUATIONS = unicode("[]{}<>)(^~_-‐;!¡|?¿/·\"«»“”„–:,.▶*+'’。，、､‚｡"
        "；：「」『』（）？！─…《》〈〉．～،؟।—‧", "utf-8")
# XXX added normal apostrophe
APOSTROPHES = re.compile(u'[\u2019\\"´\u2018\u0060\u2032\u02BB\u0301\u0300]')

PUNCTUATIONS_NO_APOSTROPHE = PUNCTUATIONS.replace("'", '')

PUNCTUATIONS_NO_APOSTROPHE_NO_COMPACT_FORM = PUNCTUATIONS_NO_APOSTROPHE.replace(u'[', u'').replace(
        u']', u'').replace(u'/', u'')

# We define the set of accepted characters as alphanumeric characters along with a selection of
# punctuation/symbols that we consider important to the meaning of words, or part of a word
# (for example, $ before a price, and apostrophes in words like I'm).
# Hyphens are not included since we strip them separately in the normalization process.
# This set unfortunately includes the underscore, since '\w' matches underscores
ACCEPTED_SYMBOLS = u"'$%&€"
# Matches unicode characters in categories L* and N* in python. Different behavior in client,
# where Swift matches unicode characters in categories Ll, Lu, Lt, Lo, Nd
ALPHANUMERIC_CHARACTERS = u"\\w"
ACCEPTED_CHARACTERS = u"{}{}".format(re.escape(ACCEPTED_SYMBOLS),
        ALPHANUMERIC_CHARACTERS)

# XXX This is a modified version of the server's local grading normalization rules.
# We changed this to strip all apostrophes.
# Normalization rules that are used in all languages.  Each element is a (regex, replacement)
# pair.  See get_normalization_data().
GLOBAL_NORMALIZATION_RULES = [
        # Make all apostrophes identical.
        (unicode(APOSTROPHES.pattern), u"'"),

        # Strip hyphens. The monolith will make edges in the graph such as
        # {'orig': 'est-ce', 'lenient': 'estce'} and the iOS client needs to match those when gathering
        # suggestions. An alternative might be to also look at the `orig`.
        (u'[-]', ''),

        # Treat underscore like whitespace (it is included in the ACCEPTED_CHARACTERS set, but ideally
        # should not be)
        (u'[_]', ' '),

        # Treat rest of non-accepted characters as whitespace.
        (u'[^{}]'.format(ACCEPTED_CHARACTERS), u' '),

        # Replace one or more consecutive whitespace characters with a single space.
        (u'\\s+', u' '),
        # Remove leading and trailing whitespace.
        (u'^ | $', ''),
        ]

# The global normalization rules are best suited for reducing a string to a canonical
# representation for grading and evaluation. This set of rules retains punctuation that for many
# languages, is important to the legibility of a sentence. For example, the global normalization
# rules would reduce 'est-ce' to 'estce' by design. This is acceptable for grading, but not
# appropriate for display to the user in a place such as the better answer text.
PRINTABLE_NORMALIZATION_RULES = [
        # Make all apostrophes identical.
        (unicode(APOSTROPHES.pattern), u"'"),
        # Treat punctuation as whitespace.
        (u'[%s]' % re.escape(IGNORED_PUNCTUATION_FRENCH), ' '),
        # Replace one or more consecutive whitespace characters with a single space.
        (u'\\s+', u' '),
        # Remove leading and trailing whitespace.
        (u'^ | $', ''),
        ]

# XXX This is a modified version of the server's local grading accented character map.
ACCENTED_CHARACTER_MAP = {
        u'Ă': u'A', u'Ć': u'C', u'Ċ': u'C', u'Ď': u'D',
        u'Ē': u'E', u'Ė': u'E', u'Ě': u'E', u'Ğ': u'G',
        u'Ģ': u'G', u'Ħ': u'H', u'Ī': u'I', u'Į': u'I',
        u'Ķ': u'K', u'ĺ': u'l', u'ľ': u'l', u'Ã': u'A',
        u'ł': u'l', u'Ç': u'C', u'ņ': u'n', u'Ë': u'E',
        u'Ï': u'I', u'Ŏ': u'O', u'Ó': u'O', u'×': u'x',
        u'Ŗ': u'R', u'Û': u'U', u'Ś': u'S', u'Ş': u'S',
        u'ã': u'a', u'Ţ': u'T', u'ç': u'c', u'Ŧ': u'T',
        u'ë': u'e', u'Ū': u'U', u'ï': u'i', u'Ů': u'U',
        u'ó': u'o', u'Ų': u'U', u'÷': u'/', u'Ŷ': u'Y',
        u'û': u'u', u'ź': u'z', u'ÿ': u'y', u'ž': u'z',
        u'ā': u'a', u'ą': u'a', u'ĉ': u'c', u'č': u'c',
        u'đ': u'd', u'ĕ': u'e', u'ę': u'e', u'ĝ': u'g',
        u'ġ': u'g', u'ĥ': u'h', u'ĩ': u'i', u'ĭ': u'i',
        u'ı': u'i', u'ĵ': u'j', u'Ĺ': u'L', u'Ľ': u'L',
        u'Ł': u'L', u'À': u'A', u'Ņ': u'N', u'Ä': u'A',
        u'È': u'E', u'ō': u'o', u'Ì': u'I', u'ő': u'o',
        u'Ð': u'D', u'ŕ': u'r', u'Ô': u'O', u'ř': u'r',
        u'Ø': u'O', u'ŝ': u's', u'Ü': u'U', u'š': u's',
        u'à': u'a', u'ť': u't', u'ä': u'a', u'ũ': u'u',
        u'è': u'e', u'ŭ': u'u', u'ì': u'i', u'ű': u'u',
        u'ð': u'd', u'ŵ': u'w', u'ô': u'o', u'Ź': u'Z',
        u'ø': u'o', u'Ž': u'Z', u'ü': u'u', u'Ā': u'A',
        u'Ą': u'A', u'Ĉ': u'C', u'Č': u'C', u'Đ': u'D',
        u'Ĕ': u'E', u'Ę': u'E', u'Ĝ': u'G', u'Ġ': u'G',
        u'Ĥ': u'H', u'Ĩ': u'I', u'Ĭ': u'I', u'İ': u'I',
        u'Ĵ': u'J', u'ĸ': u'k', u'ļ': u'l', u'Á': u'A',
        u'ŀ': u'l', u'Å': u'A', u'ń': u'n', u'É': u'E',
        u'ň': u'n', u'Í': u'I', u'Ō': u'O', u'Ñ': u'N',
        u'Ő': u'O', u'Õ': u'O', u'Ŕ': u'R', u'Ù': u'U',
        u'Ř': u'R', u'Ý': u'Y', u'Ŝ': u'S', u'á': u'a',
        u'Š': u'S', u'å': u'a', u'Ť': u'T', u'é': u'e',
        u'Ũ': u'U', u'í': u'i', u'Ŭ': u'U', u'ñ': u'n',
        u'Ű': u'U', u'õ': u'o', u'Ŵ': u'W', u'ù': u'u',
        u'Ÿ': u'Y', u'ý': u'y', u'ż': u'z', u'ă': u'a',
        u'ć': u'c', u'ċ': u'c', u'ď': u'd', u'ē': u'e',
        u'ė': u'e', u'ě': u'e', u'ğ': u'g', u'ģ': u'g',
        u'ħ': u'h', u'ī': u'i', u'į': u'i', u'ķ': u'k',
        u'Ļ': u'L', u'Ŀ': u'L', u'Ń': u'N', u'Â': u'A',
        u'Ň': u'N', u'Ê': u'E', u'ŏ': u'o', u'Î': u'I',
        u'Ò': u'O', u'ŗ': u'r', u'Ö': u'O', u'ś': u's',
        u'Ú': u'U', u'ş': u's', u'ţ': u't', u'â': u'a',
        u'ŧ': u't', u'ū': u'u', u'ê': u'e', u'ů': u'u',
        u'î': u'i', u'ų': u'u', u'ò': u'o', u'ŷ': u'y',
        u'ö': u'o', u'Ż': u'Z', u'ú': u'u'
        }


def strip_accents(string):
    """
    Apply the accented character map to a given string.
    Args:
        string (unicode): String to which to apply the accented character mapping.

    Returns:
        unicode: A string with the accented characters removed.
    """
    return u''.join((ACCENTED_CHARACTER_MAP.get(c, c) for c in string))


def normalize(string, normalization_rules=GLOBAL_NORMALIZATION_RULES):
    """
    Normalizes a string according to a list of normalization rules, executed in order. By default,
    uses the global normalization rules. See ``GLOBAL_NORMALIZATION_RULES`` for the specific
    normalization rules.

    Args:
        string (str | unicode): The string to be normalized.
        normalization_rules ([(unicode, unicode | str)]): A list of tuples whose first value is the
            regular expression to match in the string and whose second value is the replacement text
            for any matches.

    Returns:
        (str | unicode): The normalized string.
    """
    for (regex, replacement) in normalization_rules:
        string = re.sub(regex, replacement, string, flags=re.UNICODE)
    return string
