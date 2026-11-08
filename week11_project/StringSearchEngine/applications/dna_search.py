"""
dna_search.py
----------------
Part 4.2 of the string search project. Scanning a genome (or a FASTA
file of several) for known gene sequences, built on the suffix array and
suffix tree rather than KMP or Rabin-Karp specifically because of how
this application actually gets used in practice, one genome, checked
against a whole panel of many known gene sequences, over and over. That's
exactly the shape of query the suffix array and suffix tree are built
for, pay the O(n log n) or O(n^2) preprocessing cost once per genome,
then answer every gene lookup against it in O(m log n) or
O(m + occurrences) instead of re-scanning the whole genome from scratch
for every single gene in the panel.

Approximate matching, finding a gene that's present but mutated by a
substitution or two, is listed as optional in the assignment, and is
implemented here the honest, simple way, a direct sliding window Hamming
distance scan, O(n*m), rather than reaching for a proper approximate
string matching algorithm (Myers' bit-vector algorithm, or a dynamic
programming edit distance scan) that would be its own substantial
project on top of this one. The suffix structures in this project are
built for exact matching, they don't have a native notion of "close
enough", so the honest answer for "optional" approximate search is a
separate, more brute force tool rather than bolting mismatch tolerance
onto core/suffix_array.py or core/suffix_tree.py where it doesn't
belong.
"""

from core.suffix_array import build_suffix_array, sa_search
from core.suffix_tree import build_suffix_trie, substring_positions
from utils.text_loader import load_fasta, parse_fasta


def load_genome_segments(path):
    """Loads a FASTA file of one or more genome segments, returns a list
    of (header, sequence) tuples."""
    return load_fasta(path)


def load_genome_segments_from_text(fasta_text):
    """Same as load_genome_segments, for FASTA content already in memory
    as a string rather than sitting in a file on disk."""
    return parse_fasta(fasta_text)


def find_gene(genome, gene, method="sa"):
    """
    A single gene's exact occurrences in `genome`, building whichever
    index `method` needs from scratch for this one query. Fine for a
    one off lookup, scan_gene_panel below is the better choice once
    there's more than one gene to check against the same genome.
    """
    if method == "sa":
        sa = build_suffix_array(genome)
        return sa_search(genome, sa, gene)
    elif method == "tree":
        root = build_suffix_trie(genome)
        return substring_positions(genome, root, gene)
    else:
        raise ValueError(f"unknown method {method!r}, expected 'sa' or 'tree'")


def scan_gene_panel(genome, genes, method="sa"):
    """
    The realistic version of gene search, a whole panel of known gene
    sequences checked against one genome, with the genome's index built
    exactly once and reused for every gene in the panel. Returns
    {gene: [positions]}, every gene from the input appears as a key even
    if it was never found (an empty list).
    """
    if method == "sa":
        sa = build_suffix_array(genome)
        return {gene: sa_search(genome, sa, gene) for gene in genes}
    elif method == "tree":
        root = build_suffix_trie(genome)
        return {gene: substring_positions(genome, root, gene) for gene in genes}
    else:
        raise ValueError(f"unknown method {method!r}, expected 'sa' or 'tree'")


def find_gene_with_mutations(genome, gene, max_mismatches=1):
    """
    Approximate search, every position in `genome` where `gene` occurs
    allowing up to `max_mismatches` character substitutions (point
    mutations), not insertions or deletions. Returns a list of
    (position, mismatch_count) pairs, sorted by position, mismatch_count
    0 meaning an exact match was found there too.

    O(n*m) by direct Hamming distance comparison at every window, the
    simple, exact tradeoff described in this module's docstring, correct
    and easy to verify, not the asymptotically faster approach a real
    bioinformatics tool would eventually want.
    """
    n, m = len(genome), len(gene)
    if m == 0:
        raise ValueError("gene must be non-empty")
    if m > n:
        return []

    results = []
    for i in range(n - m + 1):
        mismatches = 0
        for g_char, window_char in zip(gene, genome[i:i + m]):
            if g_char != window_char:
                mismatches += 1
                if mismatches > max_mismatches:
                    break
        if mismatches <= max_mismatches:
            results.append((i, mismatches))
    return results
