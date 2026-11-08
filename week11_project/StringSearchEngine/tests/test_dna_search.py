"""
test_dna_search.py
----------------------
Tests for applications/dna_search.py: FASTA loading round trips, exact
gene search via SA and the suffix tree agree with each other and with
brute force, scanning a gene panel matches calling find_gene
individually for each gene, and the approximate (mutation tolerant)
search correctly tolerates up to its allowed mismatch count and no more.
"""

from applications.dna_search import (
    find_gene,
    find_gene_with_mutations,
    load_genome_segments_from_text,
    scan_gene_panel,
)

from conftest import brute_force_positions


class TestLoadGenomeSegments:
    def test_parses_multi_record_fasta(self):
        fasta_text = ">seqA\nACGT\nACGT\n>seqB\nTTTT\n"
        records = load_genome_segments_from_text(fasta_text)
        assert records == [("seqA", "ACGTACGT"), ("seqB", "TTTT")]


class TestFindGene:
    def test_sa_and_tree_agree_with_brute_force(self, rng):
        from utils.generators import random_dna_sequence
        genome = random_dna_sequence(300, rng=rng)
        gene = genome[50:56]
        expected = brute_force_positions(genome, gene)
        assert find_gene(genome, gene, method="sa") == expected
        assert find_gene(genome, gene, method="tree") == expected

    def test_gene_not_present(self):
        genome = "ACGTACGTACGT"
        assert find_gene(genome, "TTTT", method="sa") == []

    def test_rejects_unknown_method(self):
        import pytest
        with pytest.raises(ValueError):
            find_gene("ACGT", "AC", method="not_real")


class TestScanGenePanel:
    def test_matches_individual_find_gene_calls(self, rng):
        from utils.generators import random_dna_sequence
        genome = random_dna_sequence(500, rng=rng)
        genes = [genome[10:16], genome[100:106], "ZZZZZZ"[:4]]
        panel = scan_gene_panel(genome, genes, method="sa")
        for gene in genes:
            assert panel[gene] == find_gene(genome, gene, method="sa")

    def test_every_requested_gene_appears_as_a_key(self):
        genome = "ACGTACGTACGT"
        panel = scan_gene_panel(genome, ["ACGT", "TTTT", "GGGG"], method="tree")
        assert set(panel.keys()) == {"ACGT", "TTTT", "GGGG"}


class TestFindGeneWithMutations:
    def test_exact_match_has_zero_mismatches(self):
        genome = "AAACGTAAA"  # "ACGT" occurs at index 2
        results = find_gene_with_mutations(genome, "ACGT", max_mismatches=0)
        assert (2, 0) in results

    def test_one_substitution_is_found_within_tolerance(self):
        genome = "AAACGTAAA"
        # "ACTT" differs from "ACGT" (at genome position 2) by one character
        results = find_gene_with_mutations(genome, "ACTT", max_mismatches=1)
        assert (2, 1) in results

    def test_too_many_mismatches_is_excluded(self):
        genome = "AAACGTAAA"
        # "TTTT" differs from "ACGT" at position 2 by 3 characters
        results = find_gene_with_mutations(genome, "TTTT", max_mismatches=1)
        matching = [m for m in results if m[0] == 2]
        assert matching == []

    def test_gene_longer_than_genome_returns_empty(self):
        assert find_gene_with_mutations("AC", "ACGTACGT", max_mismatches=2) == []

    def test_rejects_empty_gene(self):
        import pytest
        with pytest.raises(ValueError):
            find_gene_with_mutations("ACGT", "", max_mismatches=1)
