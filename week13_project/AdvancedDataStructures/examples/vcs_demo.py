"""
vcs_demo.py: the Git-like version control prototype.

    python3 examples/vcs_demo.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from advds.applications.version_control import MergeConflict, MiniVCS

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def main():
    os.makedirs(OUT, exist_ok=True)
    vcs = MiniVCS(author="darius")
    vcs.write("README.md", "# Demo project\n")
    vcs.write("src/app.py", "print('v1')\n")
    c1 = vcs.commit("initial commit")

    vcs.write("src/app.py", "print('v2')\n")
    vcs.write("src/util.py", "def helper():\n    return 42\n")
    c2 = vcs.commit("add helper")

    vcs.branch("feature")
    vcs.checkout("feature")
    vcs.write("docs/guide.md", "How to use the app\n")
    vcs.commit("write guide")

    vcs.checkout("main")
    vcs.write("README.md", "# Demo project\n\nNow with a helper.\n")
    vcs.commit("update readme")
    vcs.merge("feature")

    print(vcs.graph())
    print("\nfiles at HEAD:", vcs.files())
    print("files at", c1[:7], ":", vcs.files(c1))
    d = vcs.diff(c1, c2, content=True)
    print(f"\ndiff {c1[:7]}..{c2[:7]}: added {d['added']}, modified {d['modified']} "
          f"({d['nodes_visited']} trie nodes compared)")
    print(d["patches"]["src/app.py"])

    # conflict
    vcs.branch("hotfix")
    vcs.write("src/app.py", "print('main side')\n")
    vcs.commit("main edit")
    vcs.checkout("hotfix")
    vcs.write("src/app.py", "print('hotfix side')\n")
    vcs.commit("hotfix edit")
    vcs.checkout("main")
    try:
        vcs.merge("hotfix")
    except MergeConflict as e:
        print("merge refused:", e)

    # scale: 5,000 files, one change
    big = MiniVCS()
    for i in range(5000):
        big.write(f"pkg/module_{i:04d}.py", f"# module {i}\n")
    a = big.commit("import 5000 files")
    big.write("pkg/module_1234.py", "# edited\n")
    b = big.commit("edit one file")
    t0 = time.perf_counter()
    d = big.diff(a, b)
    us = (time.perf_counter() - t0) * 1e6
    print(f"\n5,000-file repo: diff found {d['modified']} visiting {d['nodes_visited']} trie nodes in {us:.0f} us")
    print("storage:", big.storage_stats())

    vcs.plot_history(save_path=os.path.join(OUT, "vcs_history.png"))
    print("history graph saved to", os.path.join(OUT, "vcs_history.png"))


if __name__ == "__main__":
    main()
