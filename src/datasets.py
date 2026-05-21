"""Unified dataset registry. All loaders return a LabeledGraph."""

from __future__ import annotations

import csv
import gzip
import io
import tarfile
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import igraph as ig
import networkx as nx
import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

NETZ_URL = "https://networks.skewed.de/net/{name}/files/{name}.csv.zip"
SNAP = "https://snap.stanford.edu/data"


@dataclass
class LabeledGraph:
    name: str
    graph: ig.Graph
    labels: np.ndarray
    n_communities: int
    description: str

    @property
    def n_nodes(self) -> int:
        return self.graph.vcount()

    @property
    def n_edges(self) -> int:
        return self.graph.ecount()

    def __repr__(self) -> str:
        return (
            f"LabeledGraph(name={self.name!r}, n={self.n_nodes}, "
            f"m={self.n_edges}, k={self.n_communities})"
        )


def _download(url: str, dest: Path) -> Path:
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r, open(dest, "wb") as f:
        f.write(r.read())
    return dest


def _read_gz_text(path: Path) -> str:
    with gzip.open(path, "rt") as f:
        return f.read()


def _download_netz_csv(name: str) -> Path:
    cache_dir = DATA_DIR / name
    if cache_dir.exists() and (cache_dir / "nodes.csv").exists():
        return cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    url = NETZ_URL.format(name=name)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    data = urllib.request.urlopen(req).read()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(cache_dir)
    return cache_dir


def _load_netz_csv(name: str) -> tuple[ig.Graph, dict[str, list]]:
    cache_dir = _download_netz_csv(name)
    nodes: list[dict] = []
    with open(cache_dir / "nodes.csv", newline="") as f:
        reader = csv.reader(f)
        header = [h.strip().lstrip("#").strip() for h in next(reader)]
        for row in reader:
            nodes.append({h: v for h, v in zip(header, row)})

    edges: list[tuple[int, int]] = []
    with open(cache_dir / "edges.csv", newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            edges.append((int(row[0]), int(row[1])))

    g = ig.Graph(n=len(nodes), edges=edges, directed=False)
    g.simplify()
    attrs: dict[str, list] = {}
    for h in header:
        if h == "index":
            continue
        attrs[h] = [row.get(h) for row in nodes]
    if "label" in attrs:
        g.vs["name"] = attrs["label"]
    return g, attrs


def load_karate() -> LabeledGraph:
    g_nx = nx.karate_club_graph()
    nodes = list(g_nx.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    edges = [(idx[u], idx[v]) for u, v in g_nx.edges()]
    g = ig.Graph(n=len(nodes), edges=edges, directed=False)
    g.vs["name"] = [str(n) for n in nodes]
    labels = np.array(
        [0 if g_nx.nodes[n]["club"] == "Mr. Hi" else 1 for n in nodes]
    )
    return LabeledGraph("karate", g, labels, 2,
                        "Zachary's Karate Club (1977). 34 nodes, 2 factions.")


def load_dolphins() -> LabeledGraph:
    g, _ = _load_netz_csv("dolphins")
    eb = g.community_edge_betweenness(directed=False)
    labels = np.array(eb.as_clustering(n=2).membership)
    return LabeledGraph("dolphins", g, labels, 2,
                        "Lusseau Dolphins (62 nodes), k=2.")


def load_football() -> LabeledGraph:
    g, attrs = _load_netz_csv("football")
    if "value" not in attrs:
        raise RuntimeError("Football dataset missing 'value' attribute")
    labels = np.array([int(v) for v in attrs["value"]])
    return LabeledGraph("football", g, labels, int(labels.max() + 1),
                        "American College Football (Girvan & Newman 2002).")


def load_email_eu_core() -> LabeledGraph:
    edges_path = DATA_DIR / "email-Eu-core.txt.gz"
    labels_path = DATA_DIR / "email-Eu-core-department-labels.txt.gz"
    _download(f"{SNAP}/email-Eu-core.txt.gz", edges_path)
    _download(f"{SNAP}/email-Eu-core-department-labels.txt.gz", labels_path)

    node_labels: dict[int, int] = {}
    for line in _read_gz_text(labels_path).strip().splitlines():
        u, c = line.split()
        node_labels[int(u)] = int(c)
    n = max(node_labels) + 1

    edges: list[tuple[int, int]] = []
    for line in _read_gz_text(edges_path).strip().splitlines():
        u, v = line.split()
        u, v = int(u), int(v)
        if u != v:
            edges.append((u, v))

    g = ig.Graph(n=n, edges=edges, directed=False)
    g.simplify()
    components = g.connected_components()
    giant = int(np.argmax(components.sizes()))
    keep = [i for i, c in enumerate(components.membership) if c == giant]
    g_giant = g.subgraph(keep)
    labels = np.array([node_labels[i] for i in keep])
    return LabeledGraph("email-Eu-core", g_giant, labels,
                        int(len(np.unique(labels))),
                        "email-Eu-core (SNAP), giant component.")


def load_facebook_ego(ego_id: int = 1684) -> LabeledGraph:
    tar_path = DATA_DIR / "facebook.tar.gz"
    _download(f"{SNAP}/facebook.tar.gz", tar_path)

    edges_set: set[tuple[int, int]] = set()
    circles: list[list[int]] = []
    egofeat_nodes: set[int] = set()

    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            if not member.name.startswith(f"facebook/{ego_id}."):
                continue
            f = tar.extractfile(member)
            if f is None:
                continue
            text = f.read().decode()
            if member.name.endswith(".edges"):
                for line in text.strip().splitlines():
                    u, v = line.split()
                    edges_set.add((int(u), int(v)))
            elif member.name.endswith(".circles"):
                for line in text.strip().splitlines():
                    parts = line.split()
                    circles.append([int(x) for x in parts[1:]])
            elif member.name.endswith(".feat"):
                for line in text.strip().splitlines():
                    egofeat_nodes.add(int(line.split()[0]))

    if not edges_set:
        raise RuntimeError(f"No edges found for ego {ego_id}")

    all_nodes: set[int] = {ego_id}
    for u, v in edges_set:
        all_nodes.update([u, v])
    for circle in circles:
        all_nodes.update(circle)
    for node in egofeat_nodes:
        all_nodes.add(node)
        edges_set.add((ego_id, node))

    sorted_nodes = sorted(all_nodes)
    node_to_idx = {n: i for i, n in enumerate(sorted_nodes)}
    edges = [(node_to_idx[u], node_to_idx[v]) for u, v in edges_set]
    g = ig.Graph(n=len(sorted_nodes), edges=edges, directed=False)
    g.simplify()

    labels = np.full(g.vcount(), -1, dtype=int)
    for c_idx, circle in enumerate(circles):
        for node_id in circle:
            i = node_to_idx.get(node_id)
            if i is not None and labels[i] == -1:
                labels[i] = c_idx
    labels[labels == -1] = len(circles)
    unique = np.unique(labels)
    remap = {old: new for new, old in enumerate(unique)}
    labels = np.array([remap[lab] for lab in labels])

    return LabeledGraph(
        f"facebook-ego-{ego_id}", g, labels, int(len(unique)),
        f"Facebook ego {ego_id} (SNAP), {len(circles)} circles.",
    )


def load_dblp_subsample(n_top_communities: int = 500,
                        max_nodes: int = 5000) -> LabeledGraph:
    cache_path = DATA_DIR / f"dblp_top{n_top_communities}_max{max_nodes}.npz"
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        n = int(data["n"])
        edges = data["edges"].tolist()
        labels = data["labels"]
        g = ig.Graph(n=n, edges=edges, directed=False)
        return LabeledGraph(
            f"dblp-top{n_top_communities}", g, labels,
            int(len(np.unique(labels))),
            f"DBLP (SNAP), top-{n_top_communities} communities.",
        )

    ungraph_path = DATA_DIR / "com-dblp.ungraph.txt.gz"
    cmty_path = DATA_DIR / "com-dblp.top5000.cmty.txt.gz"
    _download(f"{SNAP}/bigdata/communities/com-dblp.ungraph.txt.gz", ungraph_path)
    _download(f"{SNAP}/bigdata/communities/com-dblp.top5000.cmty.txt.gz", cmty_path)

    top_communities: list[list[int]] = []
    with gzip.open(cmty_path, "rt") as f:
        for i, line in enumerate(f):
            if i >= n_top_communities:
                break
            top_communities.append([int(x) for x in line.split()])

    node_to_community: dict[int, int] = {}
    for c_idx, members in enumerate(top_communities):
        for node in members:
            node_to_community.setdefault(node, c_idx)

    seed_nodes = set(node_to_community)
    if len(seed_nodes) > max_nodes:
        kept: set[int] = set()
        for members in top_communities:
            for m in members:
                if len(kept) >= max_nodes:
                    break
                kept.add(m)
            if len(kept) >= max_nodes:
                break
        seed_nodes = kept

    node_to_idx = {n: i for i, n in enumerate(sorted(seed_nodes))}
    n = len(node_to_idx)
    edges_set: set[tuple[int, int]] = set()
    with gzip.open(ungraph_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            u, v = line.split()
            u, v = int(u), int(v)
            if u in node_to_idx and v in node_to_idx:
                ui, vi = node_to_idx[u], node_to_idx[v]
                if ui != vi:
                    edges_set.add((min(ui, vi), max(ui, vi)))

    g = ig.Graph(n=n, edges=list(edges_set), directed=False)
    g.simplify()
    idx_to_orig = {i: o for o, i in node_to_idx.items()}
    labels = np.array(
        [node_to_community[idx_to_orig[i]] for i in range(n)], dtype=int,
    )

    components = g.connected_components()
    giant = int(np.argmax(components.sizes()))
    keep = [i for i, c in enumerate(components.membership) if c == giant]
    g_giant = g.subgraph(keep)
    labels = labels[keep]
    unique = np.unique(labels)
    remap = {old: new for new, old in enumerate(unique)}
    labels = np.array([remap[lab] for lab in labels])

    np.savez_compressed(
        cache_path, n=g_giant.vcount(),
        edges=np.array(g_giant.get_edgelist(), dtype=np.int32),
        labels=labels,
    )
    return LabeledGraph(
        f"dblp-top{n_top_communities}", g_giant, labels,
        int(len(np.unique(labels))),
        f"DBLP (SNAP), top-{n_top_communities} communities.",
    )


DATASETS: dict[str, Callable[[], LabeledGraph]] = {
    "karate": load_karate,
    "dolphins": load_dolphins,
    "football": load_football,
    "email-Eu-core": load_email_eu_core,
    "facebook-ego-1684": lambda: load_facebook_ego(1684),
    "dblp-top500": lambda: load_dblp_subsample(500, max_nodes=5000),
}


def load(name: str) -> LabeledGraph:
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset {name!r}. Available: {list(DATASETS)}")
    return DATASETS[name]()


def load_many(names: list[str]) -> dict[str, LabeledGraph]:
    out: dict[str, LabeledGraph] = {}
    for name in names:
        out[name] = load(name)
    return out


def load_all() -> dict[str, LabeledGraph]:
    """Reference (small) datasets only — for tests."""
    return {n: DATASETS[n]() for n in ("karate", "dolphins", "football")}
