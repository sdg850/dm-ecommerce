import pathlib

txt = pathlib.Path('patch_notebooks.py').read_text()

# Fix n_iter: churn and ltv
txt = txt.replace("n_iter=25, scoring='recall'", "n_iter=15, scoring='recall'")
txt = txt.replace("n_iter=25, scoring='r2'", "n_iter=15, scoring='r2'")

# Fix RFM cell 12 — add sample_size to final silhouette_score
old = ("    \"work_df['RFM_Cluster'] = kmeans.fit_predict(x_scaled)\\n\"\n"
       "    \"final_sil = silhouette_score(x_scaled, work_df['RFM_Cluster'], random_state=42)\\n\"")
new = ("    \"work_df['RFM_Cluster'] = kmeans.fit_predict(x_scaled)\\n\"\n"
       "    \"SIL_SAMPLE = min(5_000, len(x_scaled))\\n\"\n"
       "    \"final_sil = silhouette_score(x_scaled, work_df['RFM_Cluster'],\\n\"\n"
       "    \"                               sample_size=SIL_SAMPLE, random_state=42)\\n\"")
txt = txt.replace(old, new)

pathlib.Path('patch_notebooks.py').write_text(txt)
print("churn n_iter=15:", txt.count("n_iter=15, scoring='recall'"))
print("ltv   n_iter=15:", txt.count("n_iter=15, scoring='r2'"))
print("sample_size in cell12:", "sample_size=SIL_SAMPLE" in txt)
