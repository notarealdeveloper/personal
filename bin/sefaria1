#!/usr/bin/env python

"""
Examples:

  sefaria gen 1 1
  sefaria Genesis 1 1
  sefaria "Genesis 1:1"
  sefaria "I Samuel" 17 4
  sefaria 1sam 17 4
  sefaria song 2 3

  sefaria gen 1 1 --lang both
  sefaria gen 1 1 --strip-niqqud
  sefaria gen 1 1 -o gen-1-1.md
  sefaria gen 1 1 --json

  # Whole book download
  sefaria gen
  sefaria genesis
  sefaria exo --lang both
  sefaria 1sam -o samuel
"""

import argparse
import json
import re
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


API_BASE = "https://www.sefaria.org/api/v3/texts"
HEBREW_MARKS_RE = re.compile(r"[\u0591-\u05BD\u05BF-\u05C7]")
PREFERRED_ENGLISH_VERSIONS = [
    "HarperCollins Study Bible, New Revised Standard Version",
    "Harper Collins Study Bible, New Revised Standard Version",
    "New Revised Standard Version",
    "NRSV",
]
SIMILAR_ENGLISH_VERSIONS = [
    "Tanakh: The Holy Scriptures, published by JPS",
    "The Contemporary Torah, Jewish Publication Society, 2006",
]


BOOKS = [
    "Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy",
    "Joshua", "Judges", "I Samuel", "II Samuel", "I Kings", "II Kings",
    "Isaiah", "Jeremiah", "Ezekiel", "Hosea", "Joel", "Amos", "Obadiah",
    "Jonah", "Micah", "Nahum", "Habakkuk", "Zephaniah", "Haggai",
    "Zechariah", "Malachi", "Psalms", "Proverbs", "Job",
    "Song of Songs", "Ruth", "Lamentations", "Ecclesiastes", "Esther",
    "Daniel", "Ezra", "Nehemiah", "I Chronicles", "II Chronicles",
]

ALIASES = {
    "gen": "Genesis",
    "exo": "Exodus",
    "ex": "Exodus",
    "lev": "Leviticus",
    "num": "Numbers",
    "deut": "Deuteronomy",
    "deu": "Deuteronomy",
    "josh": "Joshua",
    "judg": "Judges",
    "1sam": "I Samuel",
    "isam": "I Samuel",
    "1sa": "I Samuel",
    "2sam": "II Samuel",
    "iisam": "II Samuel",
    "2sa": "II Samuel",
    "1kgs": "I Kings",
    "1kings": "I Kings",
    "1ki": "I Kings",
    "2kgs": "II Kings",
    "2kings": "II Kings",
    "2ki": "II Kings",
    "isa": "Isaiah",
    "jer": "Jeremiah",
    "ezek": "Ezekiel",
    "ps": "Psalms",
    "psa": "Psalms",
    "prov": "Proverbs",
    "song": "Song of Songs",
    "sos": "Song of Songs",
    "eccl": "Ecclesiastes",
    "qoh": "Ecclesiastes",
    "lam": "Lamentations",
    "dan": "Daniel",
    "neh": "Nehemiah",
    "1chr": "I Chronicles",
    "1chron": "I Chronicles",
    "2chr": "II Chronicles",
    "2chron": "II Chronicles",
}


def norm(s):
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def resolve_book(name):
    key = norm(name)

    if key in ALIASES:
        return ALIASES[key]

    exact = [b for b in BOOKS if norm(b) == key]
    if exact:
        return exact[0]

    matches = [b for b in BOOKS if norm(b).startswith(key)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise SystemExit(f"Ambiguous book prefix {name!r}: {', '.join(matches)}")

    return name


def strip_niqqud(text):
    return HEBREW_MARKS_RE.sub("", text)


def slugify(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-") or "book"


def build_ref(parts):
    if len(parts) == 1:
        return resolve_book(parts[0])

    if len(parts) >= 3 and parts[-1].replace("-", "").isdigit() and parts[-2].isdigit():
        book = resolve_book(" ".join(parts[:-2]))
        return f"{book} {parts[-2]}:{parts[-1]}"

    if len(parts) >= 2 and parts[-1].isdigit():
        book = resolve_book(" ".join(parts[:-1]))
        return f"{book} {parts[-1]}"

    return resolve_book(" ".join(parts))


def is_whole_book(ref):
    return ":" not in ref and not re.search(r"\s+\d+$", ref)


def fetch_one(ref, lang, version=None):
    params = {"return_format": "text_only"}
    if version:
        params["version"] = version
    else:
        params["version"] = {"he": "hebrew", "en": "english"}[lang]

    url = f"{API_BASE}/{quote(ref, safe='')}?{urlencode(params)}"
    req = Request(url, headers={"User-Agent": "sefaria"})

    try:
        with urlopen(req, timeout=30) as r:
            return json.load(r)
    except HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise SystemExit(f"Sefaria HTTP error {e.code}: {body}") from e
    except URLError as e:
        raise SystemExit(f"Sefaria request failed: {e.reason}") from e


def available_versions(data, lang=None):
    versions = data.get("available_versions") or []
    if lang:
        versions = [v for v in versions if v.get("language") == lang]
    return versions


def english_versions(data):
    return [
        v
        for v in available_versions(data, "en")
        if v.get("actualLanguage", "en") == "en"
        or v.get("languageFamilyName") == "english"
    ]


def find_version(data, candidates):
    versions = english_versions(data)
    by_title = {norm(v.get("versionTitle") or ""): v for v in versions}
    by_short = {norm(v.get("shortVersionTitle") or ""): v for v in versions}

    for candidate in candidates:
        key = norm(candidate)
        if key in by_title:
            return by_title[key].get("versionTitle")
        if key in by_short:
            return by_short[key].get("versionTitle")

    for candidate in candidates:
        key = norm(candidate)
        for v in versions:
            title = norm(v.get("versionTitle") or "")
            short = norm(v.get("shortVersionTitle") or "")
            if key and (key in title or key in short):
                return v.get("versionTitle")

    return None


def get_version_title(data, lang):
    for v in data.get("versions") or []:
        if v.get("language") == lang and v.get("text"):
            return v.get("versionTitle")
    return None


def default_english_version(data):
    return (
        find_version(data, PREFERRED_ENGLISH_VERSIONS)
        or find_version(data, SIMILAR_ENGLISH_VERSIONS)
    )


def fetch(ref, lang, version=None):
    if version or lang == "he":
        return fetch_one(ref, lang, version)

    if lang == "en":
        data = fetch_one(ref, lang)
        default_version = default_english_version(data)
        if default_version and default_version != get_version_title(data, "en"):
            return fetch_one(ref, lang, default_version)
        return data

    he = fetch_one(ref, "he")
    en = fetch(ref, "en")
    data = dict(he)
    data["versions"] = (he.get("versions") or []) + (en.get("versions") or [])
    return data


def format_ref(book, sections):
    if not sections:
        return book
    if len(sections) == 1:
        return f"{book} {sections[0]}"
    return f"{book} {sections[0]}:" + ":".join(str(x) for x in sections[1:])


def iter_verse_lines(text, book, sections=None, to_sections=None):
    sections = list(sections or [])
    to_sections = list(to_sections or [])

    if isinstance(text, str):
        yield format_ref(book, sections), text
        return

    if not isinstance(text, list):
        return

    if all(isinstance(x, str) for x in text):
        is_simple_range = (
            sections
            and to_sections
            and len(sections) == len(to_sections)
            and sections[:-1] == to_sections[:-1]
            and sections[-1] != to_sections[-1]
        )

        for i, line in enumerate(text, start=1):
            if is_simple_range:
                ref_sections = sections[:-1] + [sections[-1] + i - 1]
            else:
                ref_sections = sections + [i]
            yield format_ref(book, ref_sections), line
        return

    for i, item in enumerate(text, start=1):
        yield from iter_verse_lines(item, book, sections + [i], to_sections)


def get_text(data, lang):
    if data.get("error"):
        raise SystemExit(f"Sefaria error: {data['error']}")

    for v in data.get("versions") or []:
        if v.get("language") == lang and v.get("text"):
            return v["text"], v.get("versionTitle")

    raise SystemExit(f"No {lang} text returned.")


def format_lines(data, text, strip=False, base_sections=None):
    book = data.get("book") or data.get("ref", "").split()[0] or "Book"
    sections = data.get("sections") or []
    to_sections = data.get("toSections") or []

    if base_sections is not None:
        sections = base_sections
        to_sections = base_sections

    out = []
    for ref, line in iter_verse_lines(text, book, sections, to_sections):
        if strip:
            line = strip_niqqud(line)
        out.append(f"{ref}\t{line}")
    return out


def format_text(data, lang, strip_marks=False):
    sections = []

    if lang in ("he", "both"):
        text, _title = get_text(data, "he")
        sections.append("\n".join(format_lines(data, text, strip_marks)))

    if lang in ("en", "both"):
        text, _title = get_text(data, "en")
        sections.append("\n".join(format_lines(data, text)))

    return "\n".join(x for x in sections if x)


def write_book(data, outdir, lang, strip_marks=False):
    outdir.mkdir(parents=True, exist_ok=True)

    he = get_text(data, "he")[0] if lang in ("he", "both") else None
    en = get_text(data, "en")[0] if lang in ("en", "both") else None

    chapters = he or en
    if not isinstance(chapters, list):
        raise SystemExit("Whole-book request did not return chapter-level text.")

    width = max(2, len(str(len(chapters))))

    for i in range(1, len(chapters) + 1):
        sections = []

        if he is not None:
            lines = format_lines(data, he[i - 1], strip_marks, base_sections=[i])
            sections.append("\n".join(lines))

        if en is not None:
            lines = format_lines(data, en[i - 1], base_sections=[i])
            sections.append("\n".join(lines))

        path = outdir / f"{i:0{width}d}.md"
        path.write_text("\n\n".join(sections) + "\n", encoding="utf-8")

    return len(chapters)


def main():
    p = argparse.ArgumentParser(
        description="Download Hebrew and/or English text from Sefaria.",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("ref", nargs="+")
    p.add_argument("-o", "--output", type=Path)
    p.add_argument("-l", "--lang", choices=["he", "en", "both"], default="en")
    p.add_argument("-v", "--version")
    p.add_argument("-s", "--strip-niqqud", action="store_true")
    p.add_argument("--json", "--raw-json", action="store_true", dest="raw_json")

    args = p.parse_args()
    ref = build_ref(args.ref)
    data = fetch(ref, args.lang, args.version)

    if args.raw_json:
        output = json.dumps(data, ensure_ascii=False, indent=2)
        if args.output:
            args.output.write_text(output + "\n", encoding="utf-8")
        else:
            print(output)
        return 0

    if is_whole_book(ref):
        outdir = args.output or Path(slugify(ref))
        count = write_book(data, outdir, args.lang, args.strip_niqqud)
        print(f"Wrote {count} chapters to {outdir}", file=sys.stderr)
        return 0

    output = format_text(data, args.lang, args.strip_niqqud)

    if args.output:
        args.output.write_text(output + "\n", encoding="utf-8")
    else:
        print(output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
