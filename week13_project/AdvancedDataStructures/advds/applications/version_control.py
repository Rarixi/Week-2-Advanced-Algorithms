"""
version_control.py
---------------------
Part 4.2 of the AdvancedDataStructures project: a tiny Git-like version
control system built on the persistent hash trie (PersistentMap) from
trees/persistent_tree.py.

How it maps onto Git's design:

  * blobs      file contents, stored once each, addressed by SHA-1 of
               the content (identical files are stored once).
  * trees      a PersistentMap from path -> blob id. Each commit's tree
               is a new version of the map that shares every unchanged
               node with its parent's tree, so a commit that touches one
               file out of thousands costs O(log n) new nodes, not a copy.
  * commits    (tree, parents, message, author, time); the commit id is a
               SHA-1 over those fields, so history can't be rewritten
               silently.
  * branches   names pointing at commit ids; HEAD names the current branch.

Fast diffs: every trie node carries a Merkle hash of its contents, so
comparing two commits' trees descends only into subtrees whose hashes
differ. Diffing two commits that differ in k files touches O(k log n)
nodes however large the repository is (the demo measures this).

    vcs = MiniVCS()
    vcs.write("README.md", "hello\\n")
    c1 = vcs.commit("first commit")
    vcs.write("README.md", "hello world\\n")
    c2 = vcs.commit("edit readme")
    vcs.diff(c1, c2)["modified"]        # ['README.md']
    vcs.checkout(c1); vcs.read("README.md")   # 'hello\\n'

Merging is three-way at file granularity: a file changed on only one
side since the merge base is taken from that side; a file changed
differently on both sides is a conflict (reported, not auto-resolved).
"""

import difflib
import hashlib
from collections import deque

from ..trees.persistent_tree import PersistentMap


def _sha1(text):
    return hashlib.sha1(text.encode()).hexdigest()


class Commit:
    __slots__ = ("id", "tree", "parents", "message", "author", "time", "branch")

    def __init__(self, tree, parents, message, author, time):
        self.tree = tree
        self.parents = tuple(parents)
        self.message = message
        self.author = author
        self.time = time
        self.branch = None          # branch the commit was made on (for drawing; not hashed)
        self.id = _sha1(f"tree {tree.root_hash}\n" + "".join(f"parent {p}\n" for p in self.parents)
                        + f"author {author}\ntime {time}\n\n{message}")

    def short(self):
        return self.id[:7]

    def __repr__(self):
        return f"Commit({self.short()} {self.message!r})"


class MergeConflict(Exception):
    def __init__(self, paths):
        super().__init__(f"merge conflict in: {', '.join(paths)}")
        self.paths = paths


class MiniVCS:

    def __init__(self, author="student"):
        self.author = author
        self.blobs = {}                     # blob id -> content
        self.commits = {}                   # commit id -> Commit
        self.branches = {"main": None}      # branch -> commit id (None before the first commit)
        self.head = "main"                  # current branch name, or None when detached
        self.detached = None                # commit id when HEAD is detached
        self.staging = PersistentMap()      # the working tree / index
        self._clock = 0

    # -- working tree ------------------------------------------------------------

    def _store_blob(self, content):
        blob_id = _sha1(content)
        self.blobs.setdefault(blob_id, content)
        return blob_id

    def write(self, path, content):
        """Create or overwrite a file in the working tree."""
        self.staging = self.staging.set(path, self._store_blob(content))

    def remove(self, path):
        if path not in self.staging:
            raise KeyError(path)
        self.staging = self.staging.delete(path)

    def read(self, path, commit=None):
        tree = self.staging if commit is None else self._commit(commit).tree
        blob = tree.get(path)
        if blob is None:
            raise KeyError(path)
        return self.blobs[blob]

    def files(self, commit=None):
        tree = self.staging if commit is None else self._commit(commit).tree
        return tree.keys()

    # -- refs ----------------------------------------------------------------------

    def head_commit(self):
        return self.detached if self.head is None else self.branches[self.head]

    def _commit(self, ref):
        if isinstance(ref, Commit):
            return ref
        if ref in self.branches:
            ref = self.branches[ref]
        if ref in self.commits:
            return self.commits[ref]
        matches = [c for c in self.commits if c.startswith(ref)] if ref else []
        if len(matches) == 1:
            return self.commits[matches[0]]
        raise KeyError(f"unknown commit or branch {ref!r}")

    def resolve(self, ref):
        return self._commit(ref).id

    def branch(self, name, start=None):
        """Create a branch at `start` (default: HEAD)."""
        if name in self.branches:
            raise ValueError(f"branch {name!r} exists")
        self.branches[name] = self.head_commit() if start is None else self.resolve(start)
        return name

    def checkout(self, ref):
        """Switch to a branch (or detach at a commit). The working tree becomes that snapshot."""
        if ref in self.branches:
            self.head, self.detached = ref, None
            cid = self.branches[ref]
        else:
            cid = self.resolve(ref)
            self.head, self.detached = None, cid
        self.staging = self.commits[cid].tree if cid else PersistentMap()
        return cid

    # -- committing ----------------------------------------------------------------

    def commit(self, message, parents=None):
        head = self.head_commit()
        if parents is None:
            parents = [head] if head else []
        if head and not parents[1:] and self.commits[head].tree == self.staging:
            raise ValueError("nothing to commit")
        self._clock += 1
        c = Commit(self.staging, parents, message, self.author, self._clock)
        c.branch = self.head or "(detached)"
        self.commits[c.id] = c
        if self.head is None:
            self.detached = c.id
        else:
            self.branches[self.head] = c.id
        return c.id

    def status(self):
        """Changes in the working tree relative to HEAD."""
        head = self.head_commit()
        base = self.commits[head].tree if head else PersistentMap()
        d = base.diff(self.staging)
        d.pop("nodes_visited")
        return d

    # -- history -------------------------------------------------------------------

    def log(self, ref=None):
        """Commits reachable from ref (default HEAD), newest first."""
        start = self.head_commit() if ref is None else self.resolve(ref)
        if start is None:
            return []
        seen, out = set(), []
        queue = [start]
        while queue:
            queue.sort(key=lambda cid: -self.commits[cid].time)
            cid = queue.pop(0)
            if cid in seen:
                continue
            seen.add(cid)
            out.append(self.commits[cid])
            queue.extend(self.commits[cid].parents)
        return out

    def ancestors(self, ref):
        start = self.resolve(ref)
        seen = set()
        stack = [start]
        while stack:
            c = stack.pop()
            if c in seen:
                continue
            seen.add(c)
            stack.extend(self.commits[c].parents)
        return seen

    def merge_base(self, a, b):
        """Nearest common ancestor (breadth-first from b, first hit in a's ancestors)."""
        anc_a = self.ancestors(a)
        queue = deque([self.resolve(b)])
        seen = set()
        while queue:
            c = queue.popleft()
            if c in anc_a:
                return c
            if c in seen:
                continue
            seen.add(c)
            queue.extend(self.commits[c].parents)
        return None

    # -- diffs -----------------------------------------------------------------------

    def diff(self, ref_a, ref_b, content=False):
        """
        Files added / removed / modified from ref_a to ref_b, found by
        Merkle-tree comparison. 'nodes_visited' says how many trie nodes the
        comparison touched. With content=True, also unified line diffs of
        the modified files.
        """
        ta, tb = self._commit(ref_a).tree, self._commit(ref_b).tree
        d = ta.diff(tb)
        if content:
            d["patches"] = {}
            for path in d["modified"]:
                old = self.blobs[ta.get(path)].splitlines(keepends=True)
                new = self.blobs[tb.get(path)].splitlines(keepends=True)
                d["patches"][path] = "".join(difflib.unified_diff(old, new, f"a/{path}", f"b/{path}"))
        return d

    # -- merging -----------------------------------------------------------------------

    def merge(self, other, message=None):
        """
        Three-way merge of branch/commit `other` into the current branch.
        Fast-forwards when possible; raises MergeConflict listing the
        paths changed differently on both sides.
        """
        ours = self.head_commit()
        theirs = self.resolve(other)
        base = self.merge_base(ours, theirs)
        if base == theirs:
            return ours                                   # already up to date
        if base == ours:                                  # fast-forward
            if self.head is not None:
                self.branches[self.head] = theirs
            else:
                self.detached = theirs
            self.staging = self.commits[theirs].tree
            return theirs
        t_base = self.commits[base].tree if base else PersistentMap()
        t_ours, t_theirs = self.commits[ours].tree, self.commits[theirs].tree
        d_ours, d_theirs = t_base.diff(t_ours), t_base.diff(t_theirs)
        changed_ours = set(d_ours["added"]) | set(d_ours["removed"]) | set(d_ours["modified"])
        changed_theirs = set(d_theirs["added"]) | set(d_theirs["removed"]) | set(d_theirs["modified"])
        conflicts = sorted(p for p in changed_ours & changed_theirs if t_ours.get(p) != t_theirs.get(p))
        if conflicts:
            raise MergeConflict(conflicts)
        merged = t_ours
        for p in changed_theirs - changed_ours:
            blob = t_theirs.get(p)
            merged = merged.delete(p) if blob is None else merged.set(p, blob)
        self.staging = merged
        name = other if isinstance(other, str) else self._commit(other).short()
        return self.commit(message or f"Merge {name} into {self.head or 'HEAD'}", parents=[ours, theirs])

    # -- storage statistics ----------------------------------------------------------

    def storage_stats(self):
        """
        Trie nodes actually stored across all commits (shared nodes counted
        once) vs. what full per-commit copies of the tree would need.
        """
        all_ids = set()
        full = 0
        for c in self.commits.values():
            ids = c.tree.node_ids()
            full += len(ids)
            all_ids |= ids
        return {"commits": len(self.commits), "blobs": len(self.blobs),
                "tree_nodes_stored": len(all_ids), "tree_nodes_if_copied": full,
                "sharing_factor": full / max(len(all_ids), 1)}

    # -- history visualization (optional deliverable) ------------------------------

    def graph(self):
        """
        Text history graph, newest first, one line per commit:
        `* 1a2b3c4 (main) message`, with merge commits marked `M`.
        """
        labels = {}
        for name, cid in self.branches.items():
            if cid:
                labels.setdefault(cid, []).append(name)
        lines = []
        for c in sorted(self.commits.values(), key=lambda c: -c.time):
            mark = "M" if len(c.parents) > 1 else "*"
            refs = f" ({', '.join(sorted(labels[c.id]))})" if c.id in labels else ""
            parents = " <- " + ", ".join(self.commits[p].short() for p in c.parents) if c.parents else ""
            lines.append(f"{mark} {c.short()}{refs} {c.message}{parents}")
        return "\n".join(lines)

    def plot_history(self, save_path=None):
        """Commit DAG drawn with matplotlib: time on x, one lane per branch the commit was made on."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        commits = sorted(self.commits.values(), key=lambda c: c.time)
        made_on = {c.branch for c in commits}
        order = sorted(made_on, key=lambda b: (b != "main", b))
        lane = {c.id: order.index(c.branch) for c in commits}
        fig, ax = plt.subplots(figsize=(max(6, len(commits) * 0.9), 1.0 + 0.8 * len(order)))
        colors = ["#2a6fdb", "#e0703a", "#2f9e6e", "#8e5bd0", "#c23b5a"]
        for c in commits:
            for p in c.parents:
                pc = self.commits[p]
                ax.plot([pc.time, c.time], [lane[p], lane[c.id]], color="#9a9a9a", lw=1.2, zorder=1)
        for c in commits:
            ax.scatter([c.time], [lane[c.id]], s=160, color=colors[lane[c.id] % len(colors)], zorder=2,
                       marker="D" if len(c.parents) > 1 else "o")
            ax.annotate(f"{c.short()}\n{c.message[:18]}", (c.time, lane[c.id]), xytext=(0, 12),
                        textcoords="offset points", ha="center", fontsize=7)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order)
        ax.set_xlabel("commit order")
        ax.set_ylim(-0.6, len(order) - 0.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title("Commit history (diamond = merge)", loc="left", fontsize=10)
        if save_path:
            fig.savefig(save_path, dpi=120, bbox_inches="tight")
        return fig
