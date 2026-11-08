"""
text_loader.py
------------------
Part of the string search project. Loading large text files for the
search engine and benchmark suite to chew on, plus parsing the one other
text format this project needs to read, FASTA, the standard format
genome segments ship in for dna_search.py.

Kept deliberately simple, this project's own benchmark only goes up to a
few megabytes of text (the 100MB tier the assignment lists as optional
was skipped), so there's no need for a streaming parser here, reading a
file fully into memory and handing back a plain string is both correct
and fast enough.
"""

import os


def load_text(path, encoding="utf-8"):
    """Reads an entire text file into a single string."""
    with open(path, "r", encoding=encoding) as f:
        return f.read()


def load_text_bytes_size(path):
    """Returns a file's size in bytes without reading its contents,
    used by the benchmark suite to label dataset size tiers accurately
    rather than trusting a hardcoded KB/MB figure."""
    return os.path.getsize(path)


def parse_fasta(text):
    """
    Parses FASTA formatted text (either already loaded as a string, or
    read from a file with load_text first) into a list of (header,
    sequence) tuples. FASTA records start with a '>' header line,
    followed by one or more lines of sequence data that get concatenated
    together (and have any whitespace stripped, since FASTA files are
    often wrapped at a fixed line width like 60 or 70 characters).

    Lines before the first '>' are ignored. An empty or whitespace-only
    input returns an empty list rather than raising.
    """
    records = []
    header = None
    sequence_parts = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(sequence_parts)))
            header = line[1:].strip()
            sequence_parts = []
        else:
            sequence_parts.append(line)

    if header is not None:
        records.append((header, "".join(sequence_parts)))

    return records


def load_fasta(path, encoding="utf-8"):
    """Convenience wrapper, loads a FASTA file and parses it in one
    call."""
    return parse_fasta(load_text(path, encoding=encoding))


def write_fasta(records, path):
    """Writes a list of (header, sequence) tuples back out as a FASTA
    file, wrapping sequence lines at 70 characters the way most real
    FASTA files are formatted. Mainly used by tests and the benchmark
    suite to materialize synthetic genomes generated in memory."""
    with open(path, "w") as f:
        for header, sequence in records:
            f.write(f">{header}\n")
            for i in range(0, len(sequence), 70):
                f.write(sequence[i:i + 70] + "\n")
