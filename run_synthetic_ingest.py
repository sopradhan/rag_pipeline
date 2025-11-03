"""Run the synthetic generator and ingest the produced data via DataIngestor.

This script demonstrates using `sample_schema.yaml` -> synthetic_data -> DataIngestor
to ensure the ingestion path works.
"""
from synthetic_data import generate_from_yaml
from data_ingest import DataIngestor
import os


def main():
    here = os.path.dirname(__file__)
    schema_path = os.path.join(here, 'sample_schema.yaml')
    print('Loading schema:', schema_path)
    df = generate_from_yaml(schema_path)
    print('Generated DataFrame shape:', df.shape)
    print(df.head())

    # Convert to list-of-dicts and run through DataIngestor json loader
    records = df.to_dict(orient='records')
    di = DataIngestor()
    X = di.ingest('json', payload=records)
    print('Ingested features shape (numeric columns):', X.shape)

    # Optionally save a CSV for inspection
    out_csv = os.path.join(here, 'synthetic_output.csv')
    df.to_csv(out_csv, index=False)
    print('Wrote sample CSV to', out_csv)


if __name__ == '__main__':
    main()
