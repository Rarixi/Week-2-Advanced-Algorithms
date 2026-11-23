"""Range query engine, mini version control, DNA index."""
import random

import pytest

from advds.applications.dna_index import (DNAIndex, naive_count, parse_fasta,
                                          random_genome, suffix_array, write_fasta,
                                          read_fasta)
from advds.applications.range_queries import RangeQueryEngine
from advds.applications.version_control import MergeConflict, MiniVCS


# --- range query engine ------------------------------------------------------

def test_engine_random_with_time_travel():
    r = random.Random(5)
    a = [r.randint(0, 100) for _ in range(64)]
    eng = RangeQueryEngine(a)
    snaps = {0: list(a)}
    t = 0
    for _ in range(300):
        t += r.randint(1, 3)
        l = r.randrange(64)
        rr = r.randrange(l, 64)
        v = r.randint(-10, 10)
        if r.random() < 0.5:
            a[l] = v
            eng.update(l, v, t=t)
        else:
            for i in range(l, rr + 1):
                a[i] += v
            eng.range_add(l, rr, v, t=t)
        snaps[t] = list(a)
        s = a[l:rr + 1]
        assert (eng.sum(l, rr), eng.min(l, rr), eng.max(l, rr)) == (sum(s), min(s), max(s))
    assert eng.verify()
    for tt in range(0, t + 1, 7):
        ref = snaps[max(k for k in snaps if k <= tt)]
        assert eng.snapshot(time=tt) == ref
        assert eng.sum_at(3, 40, time=tt) == sum(ref[3:41])
        assert eng.min_at(0, 63, time=tt) == min(ref)


def test_engine_doc_example_and_undo():
    eng = RangeQueryEngine([4, 8, 15, 16, 23, 42])
    eng.update(2, 100, t=10)
    eng.range_add(0, 5, 1, t=20)
    assert eng.sum(0, 5) == 199 and eng.sum_at(0, 5, time=15) == 193
    assert eng.changes_between(time_a=5, time_b=25)[2] == (2, 15, 101)
    eng.undo()
    assert eng.to_list() == [4, 8, 100, 16, 23, 42] and eng.verify()
    eng.add(0, 5)                        # history branches from the rolled-back version
    assert eng.get(0) == 9 and eng.num_versions == 4
    with pytest.raises(ValueError):
        eng.update(0, 1, t=1)            # time can't go backwards


def test_engine_rollback_by_time():
    eng = RangeQueryEngine([0] * 10)
    for t in range(1, 11):
        eng.add(t - 1, t, t=t)
    eng.rollback(time=4)
    assert eng.to_list() == [1, 2, 3, 4, 0, 0, 0, 0, 0, 0] and eng.verify()


# --- version control -------------------------------------------------------------

def _repo():
    v = MiniVCS()
    v.write("README.md", "hello\n")
    c1 = v.commit("first")
    v.write("README.md", "hello world\n")
    v.write("a.py", "x = 1\n")
    c2 = v.commit("second")
    return v, c1, c2


def test_commit_checkout_and_old_versions():
    v, c1, c2 = _repo()
    assert v.files(c1) == ["README.md"] and v.read("README.md", c1) == "hello\n"
    v.checkout(c1)
    assert v.read("README.md") == "hello\n" and v.head is None
    v.checkout("main")
    assert v.read("a.py") == "x = 1\n"
    with pytest.raises(ValueError):
        v.commit("nothing changed")


def test_diff_with_patches():
    v, c1, c2 = _repo()
    d = v.diff(c1, c2, content=True)
    assert (d["added"], d["removed"], d["modified"]) == (["a.py"], [], ["README.md"])
    assert "-hello\n+hello world" in d["patches"]["README.md"]
    assert v.diff(c2, c1)["removed"] == ["a.py"]


def test_status_and_remove():
    v, _, _ = _repo()
    v.remove("a.py")
    v.write("b.py", "b")
    assert v.status() == {"added": ["b.py"], "removed": ["a.py"], "modified": []}
    with pytest.raises(KeyError):
        v.remove("nope")


def test_branch_merge_and_graph(tmp_path):
    v, c1, c2 = _repo()
    v.branch("feature")
    v.checkout("feature")
    v.write("b.py", "b\n")
    v.commit("add b")
    v.checkout("main")
    v.write("a.py", "x = 2\n")
    v.commit("change a")
    m = v.merge("feature")
    assert len(v.commits[m].parents) == 2
    assert v.files() == ["README.md", "a.py", "b.py"] and v.read("a.py") == "x = 2\n"
    assert v.graph().startswith("M ")
    assert len(v.log()) == 5
    v.plot_history(save_path=str(tmp_path / "h.png"))
    assert (tmp_path / "h.png").exists()


def test_fast_forward_and_conflict():
    v, _, c2 = _repo()
    v.branch("ff")
    v.checkout("ff")
    v.write("c.py", "c")
    tip = v.commit("c")
    v.checkout("main")
    assert v.merge("ff") == tip and v.branches["main"] == tip
    v.branch("x")
    v.write("a.py", "main side\n")
    v.commit("m")
    v.checkout("x")
    v.write("a.py", "x side\n")
    v.commit("x")
    v.checkout("main")
    with pytest.raises(MergeConflict) as e:
        v.merge("x")
    assert e.value.paths == ["a.py"]


def test_large_repo_diff_is_cheap_and_storage_shared():
    v = MiniVCS()
    for i in range(2000):
        v.write(f"src/module_{i}.py", f"# module {i}\n")
    c1 = v.commit("import")
    v.write("src/module_42.py", "# changed\n")
    c2 = v.commit("one change")
    d = v.diff(c1, c2)
    assert d["modified"] == ["src/module_42.py"] and d["nodes_visited"] < 100   # trie has ~2,100 nodes
    st = v.storage_stats()
    assert st["sharing_factor"] > 1.9


def test_identical_content_stored_once():
    v = MiniVCS()
    v.write("a", "same")
    v.write("b", "same")
    v.commit("dup")
    assert len(v.blobs) == 1


# --- DNA index ------------------------------------------------------------------

@pytest.fixture(scope="module")
def genome():
    seq, planted = random_genome(30000, seed=1)
    return seq, planted, DNAIndex(seq)


def test_suffix_array_matches_sorting():
    rng = random.Random(0)
    for n in (1, 2, 10, 200):
        s = [rng.randint(1, 3) for _ in range(n)] + [0]
        sa = suffix_array(s).tolist()
        assert sa == sorted(range(len(s)), key=lambda i: s[i:])


def test_range_counts_and_quantiles(genome):
    seq, _, idx = genome
    r = random.Random(2)
    for _ in range(300):
        l = r.randrange(len(seq))
        rr = r.randrange(l, len(seq))
        sub = seq[l:rr + 1]
        for b in "ACGT":
            assert idx.count(b, l, rr) == sub.count(b)
        k = r.randrange(len(sub))
        assert idx.kth_smallest(l, rr, k) == sorted(sub)[k]
    assert idx.base_at(1234) == seq[1234]
    assert idx.gc_content(0, len(seq) - 1) == pytest.approx((seq.count("G") + seq.count("C")) / len(seq))


def test_pattern_counts_and_positions(genome):
    seq, planted, idx = genome
    for motif, pos in planted.items():
        assert idx.pattern_count(motif) == naive_count(seq, motif)
        found = idx.find(motif)
        assert set(pos) <= set(found)
        assert all(seq.startswith(motif, p) for p in found)
    for p in ("A", "CG", "ACGTA", "TTTTTTTTTT", seq[500:520], "GATTACAX"):
        assert idx.pattern_count(p) == naive_count(seq, p)
        assert idx.find(p) == [i for i in range(len(seq)) if seq.startswith(p, i)]


def test_n_bases_and_window_profile():
    idx = DNAIndex("ACGTNNACGTacgt")
    assert idx.composition(0, 13) == {"A": 3, "C": 3, "G": 3, "N": 2, "T": 3}
    assert idx.gc_content(0, 13) == 0.5
    assert idx.find("ACGT") == [0, 6, 10]
    assert idx.window_profile("A", 7) == [(0, 2), (7, 1)]


def test_space(genome):
    seq, _, idx = genome
    rep = idx.size_report()
    assert rep["wavelet_payload_bits_per_base"] == 2.0
    assert rep["wavelet_tree_bits"] < rep["plain_text_bits"] / 2
    assert rep["fm_index_bits"] < rep["naive_suffix_array_bits"] / 5


def test_fasta_round_trip(tmp_path):
    recs = {"chr1_frag": "ACGT" * 40, "chr2_frag": "GGCC" * 10}
    write_fasta(tmp_path / "x.fa", recs, width=50)
    assert read_fasta(tmp_path / "x.fa") == recs
    assert parse_fasta(">a desc\nacg\nRYT\n") == {"a": "ACGRYT"}
