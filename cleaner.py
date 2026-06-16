#!/usr/bin/env python3
"""
data_cleaner.py — Clean and prepare email Parquet files for LLM classification.

Usage:
    python data_cleaner.py --input <path_to_file_or_dir> --output <path> [--csv]

Output columns:
    archiveMessageId  — unchanged from input
    cleaned_text      — fully cleaned, structured, and redacted email text
"""

import re
import argparse
import html
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from bs4 import BeautifulSoup, Comment


# ─────────────────────────────────────────────────────────────────────────────
# 1. HTML / CSS Cleaning
# ─────────────────────────────────────────────────────────────────────────────

# Matches email addresses written as <user@domain.tld> (common in email headers
# and "On ... wrote:" lines). These look like HTML tags to BeautifulSoup, so we
# protect them before parsing and restore them afterward.
_ANGLE_EMAIL_RE = re.compile(
    r"<([A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})>"
)
_EMAIL_TOKEN_RE = re.compile(r"__AEMAIL__([^_]+)__ENDAEMAIL__")


def clean_html(text: str) -> str:
    """
    Strip all HTML/CSS markup from a string.

    - Protects <email@addr> patterns before parsing so BeautifulSoup does not
      consume them as HTML tags (they are restored after extraction)
    - Removes <script> and <style> blocks entirely (including their contents)
    - Strips HTML comments
    - Converts HTML entities (&amp; &nbsp; etc.) to plain-text equivalents
    - Falls back to html.parser if lxml is unavailable
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Protect angle-bracketed email addresses before BS4 parses the markup
    text = _ANGLE_EMAIL_RE.sub(r"__AEMAIL__\1__ENDAEMAIL__", text)

    try:
        soup = BeautifulSoup(text, "lxml")
    except Exception:
        soup = BeautifulSoup(text, "html.parser")

    for tag in soup(["script", "style"]):
        tag.decompose()

    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        comment.extract()

    plain = soup.get_text(separator="\n")
    plain = html.unescape(plain)

    # Restore protected email addresses
    plain = _EMAIL_TOKEN_RE.sub(r"<\1>", plain)
    return plain


# ─────────────────────────────────────────────────────────────────────────────
# 2. Whitespace Normalization
# ─────────────────────────────────────────────────────────────────────────────

_UNICODE_SPACE_RE = re.compile(r"[\u00a0\u2000-\u200b\u202f\u205f\u3000\t\r]")
_MULTI_BLANK_RE   = re.compile(r"\n{3,}")
_MULTI_SPACE_RE   = re.compile(r" {2,}")


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace throughout a string:
    - Replace non-breaking / Unicode whitespace variants with regular spaces
    - Strip leading/trailing whitespace from each line
    - Collapse 3+ consecutive blank lines into one
    - Collapse repeated spaces within a line
    """
    if not isinstance(text, str):
        return ""

    text = _UNICODE_SPACE_RE.sub(" ", text)
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(lines)
    text = _MULTI_BLANK_RE.sub("\n\n", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# 3. Sensitive Data Redaction
# ─────────────────────────────────────────────────────────────────────────────

# Email first — avoids phone-number regex partial-matching the numeric local part
_REDACTION_PATTERNS: list[tuple[re.Pattern, str]] = [
    (
        re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
        "[EMAIL REDACTED]",
    ),
    (
        re.compile(
            r"(\+?1[\s.\-]?)?"          # optional +1 country code
            r"(\(?\d{3}\)?[\s.\-]?)"    # area code
            r"\d{3}[\s.\-]?\d{4}"       # local number
            r"(\s?(ext|x|ext\.)\s?\d+)?",  # optional extension
            re.IGNORECASE,
        ),
        "[PHONE REDACTED]",
    ),
    (
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "[SSN REDACTED]",
    ),
    (
        # 13–19 digits, optional spaces or dashes between digit groups
        re.compile(r"\b(?:\d[ \-]?){13,19}\b"),
        "[CC REDACTED]",
    ),
    (
        re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
        "[IP REDACTED]",
    ),
    (
        # Simplified IPv6 (full or compressed forms)
        re.compile(r"\b[0-9a-fA-F]{1,4}(?::[0-9a-fA-F]{0,4}){2,7}\b"),
        "[IP REDACTED]",
    ),
]


def redact_sensitive(text: str) -> str:
    """Replace phone numbers, SSNs, credit cards, emails, and IPs with placeholders."""
    if not isinstance(text, str):
        return ""
    for pattern, replacement in _REDACTION_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


# ─────────────────────────────────────────────────────────────────────────────
# 4. Reply Thread Parsing
# ─────────────────────────────────────────────────────────────────────────────

# "On <date>, <name> wrote:"  (Gmail / Apple Mail style)
# Allows optional leading ">" for lines nested inside quoted blocks.
# Captures the whole content between "On" and "wrote:" then splits at the
# LAST comma — this correctly handles dates that contain commas
# (e.g. "Thu, Jun 12, 2026 at 9:00 AM, Client Alice <…> wrote:").
_ON_WROTE_RE = re.compile(
    r"^[ \t>]*-*[ \t]*On\s+(?P<content>.{10,300}?)\s+wrote\s*:[ \t]*\n",
    re.MULTILINE | re.IGNORECASE,
)

# "-----Original Message-----" / "-----Forwarded Message-----" (Outlook / Lotus)
_DIVIDER_RE = re.compile(
    r"^[ \t]*[-_=*]{3,}[ \t]*"
    r"(?:Original Message|Forwarded Message|Forwarded by|Reply)"
    r"[ \t]*[-_=*]*[ \t]*$",
    re.MULTILINE | re.IGNORECASE,
)

# Outlook block header: "From: ... \n Sent: ... \n To: ..."
_OUTLOOK_HEADER_RE = re.compile(
    r"^From\s*:\s*(?P<sender>[^\n]+)\n"
    r"(?:Sent|Date)\s*:\s*(?P<date>[^\n]+)\n"
    r"To\s*:[^\n]+\n"
    r"(?:Cc\s*:[^\n]+\n)?"
    r"(?:Subject\s*:[^\n]+\n)?",
    re.MULTILINE | re.IGNORECASE,
)

# Individual field extractors (used as fallbacks inside a chunk)
_FROM_FIELD_RE = re.compile(r"^From\s*:\s*(.+)$",         re.IGNORECASE | re.MULTILINE)
_DATE_FIELD_RE = re.compile(r"^(?:Sent|Date)\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE)

# Strips leading quote markers: ">", ">>", "> > " — including space-separated nesting
_QUOTE_LINE_RE = re.compile(r"^(>[ \t]?)+", re.MULTILINE)

# Header lines to remove when cleaning a chunk body
_HEADER_LINE_RE = re.compile(
    r"^(From|To|Sent|Date|Cc|Bcc|Subject)\s*:.+$",
    re.IGNORECASE | re.MULTILINE,
)

_DATE_FORMATS = [
    # Gmail / Apple Mail: "Thu, Jun 12, 2026 at 9:00 AM"
    "%a, %b %d, %Y at %I:%M %p",
    "%a, %b %d, %Y at %H:%M",
    # Long weekday + month
    "%A, %B %d, %Y %I:%M %p",
    "%A, %B %d, %Y %H:%M",
    # Short month
    "%B %d, %Y %I:%M %p",
    "%B %d, %Y %H:%M",
    # RFC 2822
    "%a, %d %b %Y %H:%M:%S %z",
    "%a, %d %b %Y %H:%M:%S",
    "%d %b %Y %H:%M:%S %z",
    # ISO-ish
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    # US short
    "%m/%d/%Y %H:%M",
    "%m/%d/%Y %I:%M %p",
    # Date only
    "%B %d, %Y",
    "%b %d, %Y",
]


def _parse_date(raw: str) -> str:
    """Try to normalise a raw date string into YYYY-MM-DD HH:MM."""
    raw = raw.strip()
    for fmt in _DATE_FORMATS:
        try:
            dt = datetime.strptime(raw, fmt)
            return dt.strftime("%Y-%m-%d %H:%M")
        except ValueError:
            continue
    # Return the raw value untruncated (it may still be readable)
    return raw


def _strip_header_block(text: str) -> str:
    """Remove Outlook-style 'From / To / Sent / Subject' lines from a chunk."""
    return _HEADER_LINE_RE.sub("", text).strip()


class _Boundary:
    """A detected thread-break position with optional sender/date metadata."""
    __slots__ = ("start", "end", "sender", "date")

    def __init__(
        self,
        start: int,
        end: int,
        sender: Optional[str] = None,
        date: Optional[str] = None,
    ):
        self.start  = start
        self.end    = end
        self.sender = sender
        self.date   = date


def _find_boundaries(body: str) -> list[_Boundary]:
    """
    Locate all thread-divider positions in *body* and return them sorted by
    start position, with any captured sender/date metadata attached.
    """
    boundaries: list[_Boundary] = []
    covered: list[tuple[int, int]] = []  # ranges already claimed

    def _overlaps(start: int, end: int) -> bool:
        return any(s <= start < e or s < end <= e for s, e in covered)

    # 1. "On <date>, <name> wrote:" — highest priority (carries metadata)
    for m in _ON_WROTE_RE.finditer(body):
        if not _overlaps(m.start(), m.end()):
            content = m.group("content")
            # Split at the LAST comma to correctly separate
            # "Thu, Jun 12, 2026 at 9:00 AM" from "Client Alice <email>"
            last_comma = content.rfind(",")
            if last_comma > 5:
                date_str   = content[:last_comma].strip()
                sender_str = content[last_comma + 1:].strip()
            else:
                date_str   = None
                sender_str = content.strip()
            boundaries.append(_Boundary(
                m.start(), m.end(),
                sender=sender_str or None,
                date=_parse_date(date_str) if date_str else None,
            ))
            covered.append((m.start(), m.end()))

    # 2. Outlook block header "From: ... \n Sent: ..."
    for m in _OUTLOOK_HEADER_RE.finditer(body):
        if not _overlaps(m.start(), m.end()):
            boundaries.append(_Boundary(
                m.start(), m.end(),
                sender=m.group("sender").strip(),
                date=_parse_date(m.group("date")),
            ))
            covered.append((m.start(), m.end()))

    # 3. "-----Original Message-----" style dividers
    for m in _DIVIDER_RE.finditer(body):
        if not _overlaps(m.start(), m.end()):
            boundaries.append(_Boundary(m.start(), m.end()))
            covered.append((m.start(), m.end()))

    boundaries.sort(key=lambda b: b.start)
    return boundaries


def parse_thread(body: str) -> list[dict]:
    """
    Parse an email body into a list of message dicts, **oldest first**.

    Each dict contains:
        sender (str | None)  — extracted name / address of the sender
        date   (str | None)  — normalised to YYYY-MM-DD HH:MM when possible
        body   (str)         — plain-text message content
    """
    boundaries = _find_boundaries(body)

    if not boundaries:
        # No thread structure — single message
        clean = _QUOTE_LINE_RE.sub("", body).strip()
        return [{"sender": None, "date": None, "body": clean}]

    segments: list[dict] = []

    # Segment 0: text before the first boundary (the most recent message)
    newest_body = body[: boundaries[0].start].strip()
    segments.append({"sender": None, "date": None, "body": newest_body})

    # Remaining segments: text between consecutive boundaries
    for i, boundary in enumerate(boundaries):
        seg_start = boundary.end
        seg_end   = boundaries[i + 1].start if i + 1 < len(boundaries) else len(body)
        chunk     = body[seg_start:seg_end].strip()

        sender = boundary.sender
        date   = boundary.date

        # If the boundary didn't carry metadata, try to extract it from the chunk top
        if not sender or not date:
            from_m = _FROM_FIELD_RE.search(chunk[:500])
            date_m = _DATE_FIELD_RE.search(chunk[:500])
            if from_m and not sender:
                sender = from_m.group(1).strip()
            if date_m and not date:
                date = _parse_date(date_m.group(1))
            chunk = _strip_header_block(chunk)

        # Strip "> " quote markers from all lines
        chunk = _QUOTE_LINE_RE.sub("", chunk).strip()

        if chunk or sender or date:
            segments.append({"sender": sender, "date": date, "body": chunk})

    # Reverse so the oldest message is MESSAGE 1
    segments.reverse()
    return segments


# ─────────────────────────────────────────────────────────────────────────────
# 5. Output Formatting
# ─────────────────────────────────────────────────────────────────────────────

def format_email_block(subject: str, segments: list[dict]) -> str:
    """
    Render parsed segments into the canonical structured format:

    ---START EMAIL DATA---
    Subject: <subject>
    [MESSAGE 1]
    Sender: Name (email)
    Date: YYYY-MM-DD HH:MM
    Body: <text>
    ---END EMAIL DATA---
    """
    lines = [
        "---START EMAIL DATA---",
        f"Subject: {subject}",
    ]

    for i, seg in enumerate(segments, start=1):
        lines.append(f"[MESSAGE {i}]")
        if seg.get("sender"):
            lines.append(f"Sender: {seg['sender']}")
        if seg.get("date"):
            lines.append(f"Date: {seg['date']}")
        body = seg.get("body", "").strip()
        if body:
            lines.append(f"Body: {body}")

    lines.append("---END EMAIL DATA---")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Per-row Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def clean_row(subject: str, body: str) -> str:
    """
    Run the full cleaning pipeline for a single email.

    Steps (in order):
        1. Parse reply thread structure on the RAW body first.
           This preserves sender email addresses in <angle brackets> that
           BeautifulSoup would otherwise mistake for HTML tags.
        2. Strip HTML / CSS from each segment body individually.
        3. Normalize whitespace on subject and each segment body.
        4. Redact sensitive data (subject + each segment's sender/body).
        5. Final whitespace pass.
        6. Format into structured block.
    """
    raw_body    = body    if isinstance(body, str)    else ""
    raw_subject = subject if isinstance(subject, str) else ""

    # Step 1 — HTML cleaning (subject + body).
    # clean_html protects <email@addr> patterns before BeautifulSoup runs,
    # then restores them — so "On ... <sender@domain> wrote:" lines survive
    # HTML stripping intact and can be parsed in the next step.
    clean_body    = clean_html(raw_body)
    clean_subject = clean_html(raw_subject)

    # Step 2 — Whitespace normalisation (pre-parse)
    clean_body    = normalize_whitespace(clean_body)
    clean_subject = normalize_whitespace(clean_subject)

    # Step 3 — Thread parsing on the cleaned plain text
    segments = parse_thread(clean_body)

    # Step 4 — Redaction of sensitive data
    clean_subject = redact_sensitive(clean_subject)
    for seg in segments:
        if seg.get("sender"):
            seg["sender"] = redact_sensitive(seg["sender"])
        seg["body"] = redact_sensitive(seg.get("body", ""))

    # Step 5 — Final whitespace pass (post-redaction placeholders may shift spacing)
    for seg in segments:
        seg["body"] = normalize_whitespace(seg["body"])

    # Step 6 — Format
    return format_email_block(clean_subject, segments)


# ─────────────────────────────────────────────────────────────────────────────
# 7. DataFrame-level Orchestration
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_COLS = ["archiveMessageId", "subject", "body"]


def load_parquet(path: Path) -> pd.DataFrame:
    """
    Load one Parquet file or all *.parquet files in a directory.
    Only the three needed columns are read from disk.
    """
    if path.is_dir():
        files = sorted(path.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No .parquet files found in: {path}")
        print(f"  Found {len(files)} Parquet file(s).")
        dfs = [pd.read_parquet(f, columns=REQUIRED_COLS) for f in files]
        return pd.concat(dfs, ignore_index=True)

    return pd.read_parquet(path, columns=REQUIRED_COLS)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the full pipeline to every row of *df*.

    Returns a DataFrame with columns: archiveMessageId, cleaned_text.
    """
    initial = len(df)
    print(f"  Rows loaded:        {initial:,}")

    # Null-fill text columns before processing
    df = df.copy()
    df["subject"] = df["subject"].fillna("")
    df["body"]    = df["body"].fillna("")

    # Deduplicate on the unique email ID
    df = df.drop_duplicates(subset=["archiveMessageId"], keep="first").reset_index(drop=True)
    dupes = initial - len(df)
    print(f"  Duplicates dropped: {dupes:,}")

    total = len(df)
    print(f"  Cleaning {total:,} rows...")

    cleaned_texts: list[str] = []
    for idx, row in enumerate(df.itertuples(index=False), start=1):
        if idx % 500 == 0 or idx == total:
            print(f"    {idx:,} / {total:,}")
        cleaned_texts.append(clean_row(row.subject, row.body))

    out = pd.DataFrame({
        "archiveMessageId": df["archiveMessageId"].values,
        "cleaned_text":     cleaned_texts,
    })

    print(f"  Output rows:        {len(out):,}")
    return out


def save_output(df: pd.DataFrame, path: Path, as_csv: bool) -> None:
    """Write the cleaned DataFrame to Parquet (default) or CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if as_csv:
        df.to_csv(path, index=False)
    else:
        df.to_parquet(path, index=False)
    print(f"  Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Clean email Parquet files for LLM classification.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python data_cleaner.py --input emails.parquet
  python data_cleaner.py --input ./raw/ --output ./cleaned/output.parquet
  python data_cleaner.py --input emails.parquet --output emails_clean.csv --csv
""",
    )
    p.add_argument(
        "--input", "-i",
        required=True,
        type=Path,
        help="Path to a single .parquet file or a directory of .parquet files.",
    )
    p.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("cleaned_output.parquet"),
        help="Output file path. Default: cleaned_output.parquet",
    )
    p.add_argument(
        "--csv",
        action="store_true",
        help="Write output as CSV instead of Parquet.",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()

    out_path = args.output
    if args.csv and out_path.suffix != ".csv":
        out_path = out_path.with_suffix(".csv")

    print("=" * 54)
    print("  Email Data Cleaner")
    print("=" * 54)
    print(f"  Input:  {args.input}")
    print(f"  Output: {out_path}")
    print(f"  Format: {'CSV' if args.csv else 'Parquet'}")
    print("-" * 54)

    print("\n[1/3] Loading data...")
    df = load_parquet(args.input)

    print("\n[2/3] Cleaning data...")
    cleaned = clean_dataframe(df)

    print("\n[3/3] Saving output...")
    save_output(cleaned, out_path, as_csv=args.csv)

    print("\n" + "=" * 54)
    print("  Done.")
    print("=" * 54)


if __name__ == "__main__":
    main()
