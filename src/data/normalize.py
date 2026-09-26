import re
import unicodedata
import pandas as pd


# -------------------------------------------------------------------
# Missing-value tokens
# -------------------------------------------------------------------

MISSING_TOKENS = {
    "",
    "null",
    "<null>",
    "none",
    "<none>",
    "nan",
    "<nan>",
    "n/a",
    "na",
}


def is_missing(value: str) -> bool:
    """Return True when a value represents missing data."""

    if value is None:
        return True

    value = str(value).strip().lower()

    return value in MISSING_TOKENS


# -------------------------------------------------------------------
# Generic text normalization
# -------------------------------------------------------------------

def normalize_unicode(text: str) -> str:
    """
    Normalize Unicode without removing non-Latin scripts.

    Important:
    Hindi, Bengali, Tamil, etc. must remain intact.
    """

    return unicodedata.normalize("NFKC", str(text))


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace."""

    return " ".join(text.split())


def remove_punctuation(text: str) -> str:
    """
    Remove Unicode punctuation while preserving:

    - letters
    - combining marks
    - numbers
    - whitespace
    """

    result = []

    for char in text:
        category = unicodedata.category(char)

        if (
            category.startswith("L")
            or category.startswith("M")
            or category.startswith("N")
            or category == "Zs"
            or char.isspace()
        ):
            result.append(char)
        else:
            result.append(" ")

    return "".join(result)


# -------------------------------------------------------------------
# Legal suffix normalization
# -------------------------------------------------------------------

LEGAL_SUFFIX_PHRASES = [
    # English — longer phrases first
    "private limited",
    "private ltd",
    "pvt limited",
    "pvt ltd",
    "public limited",

    # Common legal suffixes
    "corporation",
    "incorporated",
    "limited",
    "llc",
    "llp",
    "pllc",
    "corp",
    "inc",
    "ltd",
    "pvt",
    "pc",
    "lp",
    "co",

    # Devanagari
    "प्राइवेट लिमिटेड",
    "प्रा. लि.",
    "लिमिटेड",
    "लि.",
    "एलएलपी",

    # Kannada
    "ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್",
    "ಲಿಮಿಟೆಡ್",

    # Telugu
    "ప్రైవేట్ లిమిటెడ్",
    "లిమిటెడ్",

    # Tamil
    "பிரைவேட் லிமிடெட்",
    "லிமிடெட்",

    # Bengali
    "প্রাইভেট লিমিটেড",
    "লিমিটেড",

    # Gujarati
    "પ્રાઇવેટ લિમિટેડ",
    "લિમિટેડ",

    # Malayalam
    "പ്രൈവേറ്റ് ലിമിറ്റഡ്",
    "ലിമിറ്റഡ്",

    # Odia
    "ପ୍ରାଇଭେറ്റ് ଲିମିଟେଡ୍",

    # Punjabi
    "ਪ੍ਰਾਈਵੇਟ ਲਿਮਿਟਡ",
]


# Short/alternative legal suffix forms -> canonical form
LEGAL_SUFFIX_CANONICAL = {
    "pvt ltd": "private limited",
    "private ltd": "private limited",
    "pvt limited": "private limited",

    "corp": "corporation",
    "inc": "incorporated",
    "ltd": "limited",

    "l.l.c": "llc",
    "l.l.c.": "llc",

    "l.l.p": "llp",
    "l.l.p.": "llp",

    "p.c": "pc",
    "p.c.": "pc",
}


def _normalize_suffix_text(text: str) -> str:
    """
    Normalize text specifically for legal-suffix comparison.
    """

    return normalize_whitespace(text.lower())


def find_trailing_legal_suffix(text: str) -> tuple[str, str]:
    """
    Find ONE recognized legal suffix at the END of a business name.

    Returns
    -------
    tuple[str, str]
        (base_name, canonical_suffix)

    Examples
    --------
    "bright learning company corp"
        -> ("bright learning company", "corporation")

    "holloway peak inc seafood"
        -> ("holloway peak inc seafood", "")

    "oxford infracon pvt ltd"
        -> ("oxford infracon", "private limited")
    """

    text = _normalize_suffix_text(text)

    # Check longer suffixes first.
    suffixes = sorted(
        LEGAL_SUFFIX_PHRASES,
        key=lambda x: len(_normalize_suffix_text(x)),
        reverse=True,
    )

    for raw_suffix in suffixes:
        suffix = _normalize_suffix_text(raw_suffix)

        # Entire string is the suffix.
        if text == suffix:
            return "", LEGAL_SUFFIX_CANONICAL.get(
                suffix,
                suffix,
            )

        # Suffix must be a complete trailing phrase.
        if text.endswith(" " + suffix):
            base = text[:-len(suffix)].strip()

            canonical = LEGAL_SUFFIX_CANONICAL.get(
                suffix,
                suffix,
            )

            return base, canonical

    return text, ""


def strip_trailing_legal_suffix(text: str) -> str:
    """
    Remove one recognized legal-entity suffix from the END
    of a normalized business name.

    Only trailing suffixes are removed.
    Words appearing in the middle of a business name are preserved.
    """

    base, _ = find_trailing_legal_suffix(text)

    return base


# -------------------------------------------------------------------
# Business name normalization
# -------------------------------------------------------------------

def normalize_name(name: str) -> tuple[str, str]:
    """
    Normalize a business name.

    Returns
    -------
    tuple[str, str]
        (
            suffix-stripped normalized name,
            suffix-preserved normalized name
        )
    """

    if is_missing(name):
        return "", ""

    # ---------------------------------------------------------------
    # Unicode normalization
    # ---------------------------------------------------------------

    text = normalize_unicode(name)

    # Lowercase while preserving Unicode scripts.
    text = text.lower()

    # ---------------------------------------------------------------
    # Standardize ampersand
    # ---------------------------------------------------------------

    text = text.replace("&", " and ")

    # ---------------------------------------------------------------
    # Remove Unicode punctuation while preserving:
    # - letters
    # - combining marks
    # - numbers
    # - whitespace
    # ---------------------------------------------------------------

    text = remove_punctuation(text)

    # Collapse whitespace.
    text = normalize_whitespace(text)

    # ---------------------------------------------------------------
    # Normalize common English abbreviations/suffixes
    # ---------------------------------------------------------------

    replacement_map = {
        "pvt ltd": "private limited",
        "pvt. ltd.": "private limited",
        "pvt ltd.": "private limited",
        "pvt. ltd": "private limited",

        "corp": "corporation",
        "inc": "incorporated",
        "ltd": "limited",

        "llc": "llc",
        "l.l.c": "llc",
        "l.l.c.": "llc",

        "llp": "llp",
        "l.l.p": "llp",
        "l.l.p.": "llp",

        "pc": "pc",
        "p.c": "pc",
        "p.c.": "pc",
    }

    # Replace phrase variants.
    for old, new in sorted(
        replacement_map.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        pattern = rf"(?<!\w){re.escape(old)}(?!\w)"

        text = re.sub(
            pattern,
            new,
            text,
            flags=re.IGNORECASE,
        )

    text = normalize_whitespace(text)

    # ---------------------------------------------------------------
    # IMPORTANT:
    # Detect the legal suffix ONLY at the END.
    #
    # Example:
    #
    # "holloway peak inc seafood"
    #
    # remains unchanged because "inc" is in the middle.
    #
    # But:
    #
    # "holloway peak seafood inc"
    #
    # becomes:
    #
    # "holloway peak seafood"
    # ---------------------------------------------------------------

    base, canonical_suffix = find_trailing_legal_suffix(text)

    # ---------------------------------------------------------------
    # Preserve complete normalized name with canonical suffix.
    # ---------------------------------------------------------------

    if canonical_suffix:
        if base:
            with_suffix = f"{base} {canonical_suffix}"
        else:
            with_suffix = canonical_suffix

        clean = base
    else:
        clean = text
        with_suffix = text

    return clean, with_suffix


# -------------------------------------------------------------------
# Address normalization
# -------------------------------------------------------------------

ADDRESS_ABBREVIATIONS = {
    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "av": "avenue",
    "blvd": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "ct": "court",
    "hwy": "highway",
    "pkwy": "parkway",
    "pl": "place",
    "trl": "trail",
    "ter": "terrace",
    "cir": "circle",
}


def extract_landmark(address: str) -> tuple[str, str]:
    """
    Extract simple landmark phrases such as:

        Near Apollo Hospital
        Opposite City Mall

    Returns
    -------
    tuple[str, str]
        (cleaned address, landmark)
    """

    if is_missing(address):
        return "", ""

    text = normalize_unicode(address)
    text = text.lower()

    landmark = ""

    patterns = [
        r"\b(near|opposite|opp\.?)\s+([^,]+)",
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            text,
            flags=re.UNICODE,
        )

        if match:
            landmark = match.group(0).strip()
            break

    return text, landmark


def normalize_address(address: str) -> tuple[str, str]:
    """
    Normalize an address while preserving useful information.

    Returns
    -------
    tuple[str, str]
        (
            normalized address,
            extracted landmark
        )
    """

    if is_missing(address):
        return "", ""

    text = normalize_unicode(address)
    text = text.lower()

    # Remove explicit missing-value components.
    for token in MISSING_TOKENS:
        if token:
            text = re.sub(
                rf"(?<!\w){re.escape(token)}(?!\w)",
                " ",
                text,
                flags=re.UNICODE,
            )

    # Extract landmark before punctuation normalization.
    text, landmark = extract_landmark(text)

    # Standardize common address abbreviations.
    for short, long in ADDRESS_ABBREVIATIONS.items():
        text = re.sub(
            rf"\b{re.escape(short)}\.?\b",
            long,
            text,
            flags=re.IGNORECASE,
        )

    # Normalize '&' → 'and'.
    text = text.replace("&", " and ")

    # Replace punctuation with spaces.
    text = re.sub(
        r"[^\w\s]",
        " ",
        text,
        flags=re.UNICODE,
    )

    text = normalize_whitespace(text)

    # Normalize landmark itself.
    landmark = re.sub(
        r"[^\w\s]",
        " ",
        landmark,
        flags=re.UNICODE,
    )

    landmark = normalize_whitespace(landmark)

    return text, landmark


# -------------------------------------------------------------------
# Country normalization
# -------------------------------------------------------------------

def normalize_country(country: str) -> str:
    """
    Normalize country names dynamically.

    No country filtering is performed.
    """

    if is_missing(country):
        return ""

    text = normalize_unicode(country).lower()

    text = re.sub(r"[^\w\s]", "", text)

    text = normalize_whitespace(text)

    return text



def get_known_countries(df: pd.DataFrame, country_col: str = "country") -> list[str]:
    """
    Dynamically extracts unique normalized countries from a DataFrame.
    Never hardcodes US/India.
    """
    if country_col not in df.columns:
        return []
    return sorted(list(set(normalize_country(c) for c in df[country_col].unique() if c)))



# -------------------------------------------------------------------
# Entity ID / source
# -------------------------------------------------------------------

def extract_source(entity_id: str) -> str:
    """
    Extract source prefix from IDs such as:

        S1-123456
        S2-123456
        S3-123456
    """

    if is_missing(entity_id):
        return ""

    entity_id = str(entity_id).strip()

    return entity_id.split("-", 1)[0]


# -------------------------------------------------------------------
# DataFrame-level normalization helper
# -------------------------------------------------------------------

def normalize_source(df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes an entity DataFrame and adds all Stage A normalized columns
    while preserving original raw columns intact.

    Input columns expected:
        entity_id, business_name, business_address, country

    Output columns produced:
        entity_id, source, business_name, business_name_clean,
        business_name_clean_with_suffix, business_address,
        business_address_clean, landmark, country
    """
    res = df.copy()

    if "entity_id" in res.columns:
        res["source"] = res["entity_id"].apply(extract_source)
    else:
        res["source"] = ""

    if "business_name" in res.columns:
        names_norm = res["business_name"].apply(normalize_name)
        res["business_name_clean"] = [n[0] for n in names_norm]
        res["business_name_clean_with_suffix"] = [n[1] for n in names_norm]

    if "business_address" in res.columns:
        addrs_norm = res["business_address"].apply(normalize_address)
        res["business_address_clean"] = [a[0] for a in addrs_norm]
        res["landmark"] = [a[1] for a in addrs_norm]

    if "country" in res.columns:
        res["country"] = res["country"].apply(normalize_country)

    target_order = [
        "entity_id", "source", "business_name", "business_name_clean",
        "business_name_clean_with_suffix", "business_address",
        "business_address_clean", "landmark", "country"
    ]

    out_cols = [c for c in target_order if c in res.columns]
    for c in res.columns:
        if c not in out_cols:
            out_cols.append(c)

    return res[out_cols]


# Alias for backward compatibility
normalize_dataframe = normalize_source