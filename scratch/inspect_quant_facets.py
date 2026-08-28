import pandas as pd

df = pd.read_csv("data/processed/enriched_facets.csv")

quant_facets = df[df["normalized_facet"].str.contains(r"hours|hours/week|mg/day|km/week|time/day|repetitions|cycles|sessions", case=False, na=False)]

print(f"Total quantitative facets found: {len(quant_facets)}")
for _, row in quant_facets.head(25).iterrows():
    print(f"- Name: {row['normalized_facet']}")
    print(f"  Type: {row['facet_type']} | Observable: {row['conversation_observable']}")
    print(f"  Scoring Def: {row['scoring_definition']}")
    print(f"  Score 1: {row['score_1_anchor']} | Score 5: {row['score_5_anchor']}")
    print("-" * 60)
