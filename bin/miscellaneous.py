#!/usr/bin/python2.7

import re
from porter2 import stem
from nltk.corpus import stopwords

abbrev_dict = {
    "#": "number",
    "acct": "account",
    "amt": "amount",
    "cnt": "count",
    "cust": "customer",
    "dept": "department",
    "no.": "number",
    "no": "number",
    "num": "number",
    "ord": "order",
    "pcs": "pieces",
    "qty": "quantity",
    "ref": "reference",
    "seq": "sequence",
    "shp": "ship",
    "tel": "telephone",
    "tkt": "ticket"
}

money_re = re.compile(r'^[$s]?-?(\d*([.,]\d{3})*\.\d{1,2})$|^\$-?(\d+([.,]\d{3})*)$')
number_re = re.compile(r'^-?[0-9]+([.,][0-9]+)*$')
phone_re = re.compile(r'^(((((\([0-9]{3}\))|([0-9]{3}))-?)[0-9]{3}-?)|([0-9]{3}-))[0-9]{4}$')
email_re = re.compile(r'^.+@.+\.[a-z]+$')
date_re = re.compile('|'.join([
    r'^(\d{2,4}-)?((jan)|(january)|(feb)|(february)|(mar)|(march)|(apr)|(april)|(may)|(jun)|(june)|(jul)|(july)|(aug)|(august)|(sep)|(september)|(oct)|(october)|(nov)|(november)|(dec)|(december))(-\d{2,4})?$',
    r'^((0?[1-9])|(1[0-2]))[-/.]((0?[1-9])|(1[0-9])|(2[0-9])|(3[0-1]))[-/.](20)?[0-9]{2}$',
    r'^((0?[1-9])|(1[0-9])|(2[0-9])|(3[0-1]))[-/.]((0?[1-9])|(1[0-2]))[-/.](20)?[0-9]{2}$',
    r'^(20)?[0-9]{2}[-/.]((0?[1-9])|(1[0-2]))[-/.]((0?[1-9])|(1[0-9])|(2[0-9])|(3[0-1]))$',
    r'^((monday)|(mon)|(tuesday)|(tue)|(wednesday)|(wed)|(thursday)|(thur)|(friday)|(fri)|(saturday)|(sat)|(sunday)|(sun))$'
]))
sw = stopwords.words("english")

def process_words(words):
    # Convert to lower case
    words = words.strip()
    words = words.lower()
    # replace abbreviations with full words and do word stemming
    splited = words.split(" ")
    processed_words = []
    tpes = []
    for word in splited:
        if word not in ['[addr]', '[logo]', '[supplier]']:
            word = word.strip(" ,._+=!@%^&*:;/?<>()[]{}|'").rstrip("$")
        if len(word) <= 1 and not word.isdigit() and word is not "#":
            continue
        tpe = get_type(word)
        if word in abbrev_dict:
            processed_word = stem(abbrev_dict[word])
        elif stem(word) in abbrev_dict:
            processed_word = stem(abbrev_dict[stem(word)])
        else:
            processed_word = stem(word)
        if (tpe!='text'):
            processed_word = tpe
        if processed_word not in sw:
            processed_words.append(processed_word.strip("#"))
            tpes.append(tpe)
    return (' '.join(processed_words), ','.join(tpes))

def get_type(words):
    # default type is text
    word_type = "text"
    # try parsing as date
    if words == '[addr]':
        word_type = words 
    elif words == '[logo]':
        word_type = words
    elif words == '[supplier]':
        word_type = words
    elif money_re.match(words):
        word_type = "money"
    elif number_re.match(words):
        word_type = "number"
    elif date_re.match(words):
        word_type = "date"
    elif phone_re.match(words):
        word_type = "phone"
    elif email_re.match(words):
        word_type = "email"
    # else:
        # try:
            # parse(words)
            # word_type = "date"
        # except ValueError:
            # pass
    return word_type
