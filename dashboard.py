import os
import sqlite3
import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw

st.set_page_config(page_title="GBM AI Dashboard", layout="wide", page_icon="🧬")

st.title("🧬 Glioblastoma De Novo Drug Discovery")
st.caption("Live telemetry and molecular visualization for the autonomous GitHub Actions engine.")

DB_PATH = "gbm_workspace/gbm_vault.sqlite"

if not os.path.exists(DB_PATH):
    st.warning("Database not found. Waiting for the first GitHub Action batch to complete...")
else:
    # Connect to the SQLite vault
    conn = sqlite3.connect(DB_PATH)
    
    # Fetch high-level metrics
    total_mols = pd.read_sql_query("SELECT COUNT(*) FROM discovered_molecules", conn).iloc[0, 0]
    best_score = pd.read_sql_query("SELECT MAX(total_score) FROM discovered_molecules", conn).iloc[0, 0]
    golden_mols = pd.read_sql_query("SELECT COUNT(*) FROM discovered_molecules WHERE pred_pic50 >= 8.0 AND bbb_score >= 0.75", conn).iloc[0, 0]
    
    # Display top metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Candidates Evaluated", f"{total_mols:,}")
    c2.metric("Highest Unified Score", f"{best_score:.3f}")
    c3.metric("Golden Candidates (High BBB & Potency)", f"{golden_mols:,}")
    
    st.divider()
    
    # Fetch the top 12 molecules for visualization
    top_candidates = pd.read_sql_query("""
        SELECT generation, total_score, pred_pic50, bbb_score, sa_score, smiles 
        FROM discovered_molecules 
        ORDER BY total_score DESC 
        LIMIT 12
    """, conn)
    conn.close()

    st.subheader("🏆 Top Discovered Scaffolds")
    
    # Create a visual grid of the top 12 molecules
    cols = st.columns(4)
    for index, row in top_candidates.iterrows():
        mol = Chem.MolFromSmiles(row["smiles"])
        if mol:
            # Generate a 2D image of the molecule
            img = Draw.MolToImage(mol, size=(300, 300))
            with cols[index % 4]:
                st.image(img, use_container_width=True)
                st.markdown(f"**Rank {index + 1}** (Gen {row['generation']})")
                st.markdown(f"Score: `{row['total_score']:.2f}` | pIC50: `{row['pred_pic50']:.2f}` | BBB: `{row['bbb_score']:.2f}`")
    
    st.divider()
    st.subheader("📊 Complete Leaderboard Data")
    st.dataframe(top_candidates, use_container_width=True)
