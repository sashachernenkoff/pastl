r"""
Build the gene -> subcellular compartment map consumed by
plot_egene_stratification.py (--localization_map).

Reads the human GO annotations (GAF), keeps Cellular Component (aspect "C"), and
assigns each gene one coarse compartment by walking the GO DAG: a gene belongs to
a compartment if any of its CC terms is, or descends from, that compartment's
anchor term(s). Genes with several compartments are resolved by priority
(peripheral wins), matching plot_egene_stratification._LOCALIZATION_PRIORITY.

Extracellular anchors on extracellular space + matrix (GO:0005615/0031012) rather
than the broad extracellular region (GO:0005576). Cytoplasm (GO:0005737) sits at
the bottom of the priority list and only claims genes with no more specific term.

Requires go-basic.obo and goa_human.gaf.gz (Gene Ontology downloads) in --data_dir.

Usage:
  python analysis/egene_eval/build_localization_map.py \
      --data_dir /path/to/go_files \
      --output_dir out/analysis/egene_eval
"""
import argparse
import csv
import gzip

from goatools.obo_parser import GODag

# (compartment, [anchor GO ids]) in ASSIGNMENT PRIORITY order (peripheral first).
COMPARTMENTS = [
    ("Extracellular", ["GO:0005615", "GO:0031012"]),               # extracellular space, matrix
    ("Membrane",      ["GO:0005886", "GO:0009986"]),               # plasma membrane, cell surface
    ("Secretory",     ["GO:0005783", "GO:0005794",                 # ER, Golgi,
                       "GO:0005768", "GO:0005764", "GO:0031982"]),  # endosome, lysosome, vesicle
    ("Mitochondrion", ["GO:0005739"]),
    ("Cytoskeleton",  ["GO:0005856"]),
    ("Nuclear",       ["GO:0005634"]),                             # nucleus (+ nucleoplasm, chromatin)
    ("Cytoplasm",     ["GO:0005737", "GO:0005829"]),               # cytoplasm, cytosol (catch-all)
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True,
                    help="Directory containing go-basic.obo and goa_human.gaf.gz")
    ap.add_argument("--output_dir", required=True,
                    help="Directory to write localization_symbols.tsv")
    args = ap.parse_args()

    from pathlib import Path
    data_dir = Path(args.data_dir)
    out_path = Path(args.output_dir) / "localization_symbols.tsv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    godag = GODag(str(data_dir / "go-basic.obo"))

    # Expand each anchor to itself + all descendant CC terms for fast lookup.
    comp_terms = []
    for name, anchors in COMPARTMENTS:
        terms = set()
        for a in anchors:
            if a in godag:
                terms.add(a)
                terms |= godag[a].get_all_children()   # already a set of GO id strings
        comp_terms.append((name, terms))

    def classify(go_ids: set[str]) -> str | None:
        for name, terms in comp_terms:          # priority order
            if go_ids & terms:
                return name
        return None

    gene_go: dict[str, set[str]] = {}
    with gzip.open(str(data_dir / "goa_human.gaf.gz"), "rt") as fh:
        for line in fh:
            if line.startswith("!"):
                continue
            f = line.rstrip("\n").split("\t")
            symbol, go_id, aspect = f[2], f[4], f[8]
            if aspect != "C" or go_id not in godag:   # Cellular Component only
                continue
            gene_go.setdefault(symbol.upper(), set()).add(go_id)

    rows, counts = [], {}
    for symbol, go_ids in gene_go.items():
        comp = classify(go_ids)
        if comp is None:
            continue
        rows.append((symbol, comp))
        counts[comp] = counts.get(comp, 0) + 1

    with open(out_path, "w", newline="") as out:
        w = csv.writer(out, delimiter="\t")
        w.writerow(["gene_symbol", "compartment"])
        w.writerows(sorted(rows))

    print(f"Wrote {len(rows):,} genes to {out_path}")
    for name, _ in COMPARTMENTS:
        print(f"  {name:<14} {counts.get(name, 0):>6}")


if __name__ == "__main__":
    main()
