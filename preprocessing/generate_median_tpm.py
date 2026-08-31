"""Generate a gene × tissue median TPM matrix from GTEx tissue-specific sample files."""

import argparse
import pandas as pd
import gzip


def get_symbol_mapping(gtf_path):
    """Extracts a mapping of Ensembl Gene IDs to Gene Symbols from the GTF."""
    print("Building ID-to-Symbol map from GTF...")
    mapping = {}
    with gzip.open(gtf_path, 'rt') as f:
        for line in f:
            if line.startswith('#') or '\tgene\t' not in line:
                continue
            fields = line.split('\t')[8].split(';')
            # Extract gene_id and gene_name
            d = {x.strip().split(' ')[0]: x.strip().split(' ')[1].strip('"')
                 for x in fields if len(x.strip().split(' ')) == 2}
            if 'gene_id' in d and 'gene_name' in d:
                # Store without version tags for easier matching
                mapping[d['gene_id'].split('.')[0]] = d['gene_name']
    return mapping


def generate_transcript_medians(tpm_path, attr_path, gtf_path, out_path):
    # 1. Map samples to lung tissue
    print("Mapping samples to lung tissue...")
    attr = pd.read_csv(attr_path, sep='\t')
    lung_ids = attr[attr['SMTS'] == 'Lung']['SAMPID'].tolist()

    # 2. Build Symbol Map
    symbol_map = get_symbol_mapping(gtf_path)

    # 3. Detect Header and Columns
    # Line 2 (index 2) is the header
    print("Reading header from Line 2...")
    with gzip.open(tpm_path, 'rt') as f:
        for _ in range(2): f.readline()  # skip first two lines
        header = f.readline().strip().split('\t')

    lung_cols = [c for c in lung_ids if c in header]
    # transcript_id is col 0, gene_id is col 1
    cols_to_keep = ['transcript_id', 'gene_id'] + lung_cols

    # 4. Chunked processing
    print(f"Calculating medians for {len(lung_cols)} lung samples...")
    reader = pd.read_csv(tpm_path, sep='\t', skiprows=2,
                         usecols=cols_to_keep, chunksize=10000)

    medians_list = []
    for chunk in reader:
        # Calculate median across lung samples
        chunk['Lung'] = chunk[lung_cols].median(axis=1)

        # Clean ID for mapping
        chunk['clean_gene_id'] = chunk['gene_id'].str.split('.').str[0]

        # Map IDs to Symbols (to match CPTAC targets)
        chunk['Description'] = chunk['clean_gene_id'].map(symbol_map)

        # Rename 'transcript_id' to 'Name'
        chunk = chunk.rename(columns={'transcript_id': 'Name'})

        # Keep only what the metadata script needs
        medians_list.append(chunk[['Name', 'Description', 'Lung']].dropna(subset=['Description']))

    print("Saving consolidated file...")
    final_df = pd.concat(medians_list)
    final_df.to_csv(out_path, sep='\t', index=False, compression='gzip')
    print(f"Done! Created {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tpm_gct", required=True,
                        help="GTEx transcript TPM matrix (.gct.gz)")
    parser.add_argument("--sample_attr", required=True,
                        help="GTEx sample attributes file")
    parser.add_argument("--gencode_gtf", required=True,
                        help="GENCODE annotation GTF (.gtf.gz)")
    parser.add_argument("--output_path", required=True,
                        help="Output path for the gene × tissue median TPM matrix (.txt.gz)")
    args = parser.parse_args()

    generate_transcript_medians(args.tpm_gct, args.sample_attr,
                                args.gencode_gtf, args.output_path)