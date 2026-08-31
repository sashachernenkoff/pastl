r"""
Convert ATtRACT pwm.txt + ATtRACT_db.txt to a MEME-format PWM file for FIMO.

Filters to Homo_sapiens entries only. Deduplicates by Matrix_id (first
occurrence kept). The ATtRACT pwm.txt already uses T (not U), so no
alphabet conversion is needed for FIMO scanning on DNA sequences.

ATtRACT pwm.txt format:
  >matrix_id  width
  P(A)  P(C)  P(G)  P(T)    <- one row per position
  ...

Output: {output_path} — MEME version 4 format, readable by FIMO.

Usage:
  python convert_attract.py \
      --pwm_txt   /path/to/data/motifs/attract/pwm.txt \
      --db_txt    /path/to/data/motifs/attract/ATtRACT_db.txt \
      --output    out/analysis/ism/ATtRACT_human.meme
"""
import argparse
from pathlib import Path

_OUTPUT  = "out/analysis/ism/ATtRACT_human.meme"

MEME_HEADER = """\
MEME version 4

ALPHABET= ACGT

strands: + -

Background letter frequencies
A 0.25 C 0.25 G 0.25 T 0.25

"""


def parse_db(db_path):
    """
    Parse ATtRACT_db.txt.
    Returns dict: matrix_id (str) -> gene_name (str)
    for Homo_sapiens entries only (first occurrence per matrix_id).
    """
    human_ids = {}
    with open(db_path) as f:
        header = f.readline()   # skip header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 13:
                continue
            gene_name = parts[0].strip()
            organism  = parts[3].strip()
            matrix_id = parts[11].strip()
            if organism != "Homo_sapiens":
                continue
            if matrix_id not in human_ids:
                human_ids[matrix_id] = gene_name
    print(f"  Human matrix IDs in ATtRACT_db: {len(human_ids):,}")
    return human_ids


def parse_pwm(pwm_path, keep_ids):
    """
    Parse ATtRACT pwm.txt.
    Returns dict: matrix_id -> list of rows, each row = [pA, pC, pG, pT].
    Only returns matrices whose ID is in keep_ids.
    """
    matrices = {}
    current_id = None
    current_rows = []

    with open(pwm_path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                # Save previous matrix if it's a human one
                if current_id is not None and current_id in keep_ids:
                    matrices[current_id] = current_rows
                # Parse new header: ">matrix_id\twidth"
                parts = line[1:].split("\t")
                current_id = parts[0].strip()
                current_rows = []
            else:
                vals = line.split("\t")
                if len(vals) == 4:
                    try:
                        current_rows.append([float(v) for v in vals])
                    except ValueError:
                        pass

    # Flush last matrix
    if current_id is not None and current_id in keep_ids:
        matrices[current_id] = current_rows

    print(f"  Matrices parsed from pwm.txt: {len(matrices):,}")
    return matrices


def write_meme(matrices, id_to_gene, output_path):
    """Write matrices to MEME version 4 format."""
    n_written = 0
    with open(output_path, "w") as f:
        f.write(MEME_HEADER)
        for matrix_id, rows in matrices.items():
            if not rows:
                continue
            gene_name = id_to_gene.get(matrix_id, matrix_id)
            # Sanitise name: MEME motif names must not contain spaces
            motif_name = f"M{matrix_id}"
            alt_name   = gene_name.replace(" ", "_")
            w = len(rows)
            f.write(f"MOTIF {motif_name} {alt_name}\n")
            f.write(f"letter-probability matrix: alength= 4 w= {w} nsites= 20 E= 0\n")
            for row in rows:
                f.write("  " + "  ".join(f"{v:.8f}" for v in row) + "\n")
            f.write("\n")
            n_written += 1
    print(f"  Motifs written to MEME file: {n_written:,}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pwm_txt", required=True,
                        help="ATtRACT pwm.txt")
    parser.add_argument("--db_txt",  required=True,
                        help="ATtRACT_db.txt")
    parser.add_argument("--output",  default=_OUTPUT)
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Parsing ATtRACT_db.txt: {args.db_txt}")
    id_to_gene = parse_db(args.db_txt)

    print(f"Parsing pwm.txt: {args.pwm_txt}")
    matrices = parse_pwm(args.pwm_txt, set(id_to_gene.keys()))

    print(f"Writing MEME file: {args.output}")
    write_meme(matrices, id_to_gene, out_path)
    print(f"Done -> {out_path}")


if __name__ == "__main__":
    main()
