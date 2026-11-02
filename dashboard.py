import os
import json
import sqlite3
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import py3Dmol
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors

st.set_page_config(page_title="GBM AI Dashboard", layout="wide", page_icon="🧬")

st.title("🧬 Glioblastoma De Novo Drug Discovery")
st.caption("Live telemetry, 2D/3D MMFF94s force-field modeling, and lead export for the autonomous GBM engine.")

DB_PATH = "gbm_workspace/gbm_vault.sqlite"
STATE_PATH = "gbm_workspace/adaptive_state.json"

def generate_3d_conformer_molblock(smiles):
    """Inflates 2D SMILES into a 3D MMFF94s energy-minimized structure."""
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None, None
    mol_3d = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    if AllChem.EmbedMolecule(mol_3d, params) == -1:
        AllChem.EmbedMolecule(mol_3d, useRandomCoords=True)
    
    energy = None
    props = AllChem.MMFFGetMoleculeProperties(mol_3d, mmffVariant="MMFF94s")
    if props:
        ff = AllChem.MMFFGetMoleculeForceField(mol_3d, props)
        if ff:
            ff.Minimize(maxIts=200)
            energy = round(ff.CalcEnergy(), 2)
    return Chem.MolToMolBlock(mol_3d), energy

if not os.path.exists(DB_PATH):
    st.warning("Database not found. Waiting for the first GitHub Action batch to complete...")
else:
    conn = sqlite3.connect(DB_PATH)
    vault_mols = pd.read_sql_query("SELECT COUNT(*) FROM discovered_molecules", conn).iloc[0, 0]
    best_score = pd.read_sql_query("SELECT MAX(total_score) FROM discovered_molecules", conn).iloc[0, 0]
    golden_mols = pd.read_sql_query(
        "SELECT COUNT(*) FROM discovered_molecules WHERE pred_pic50 >= 8.0 AND bbb_score >= 0.75", conn
    ).iloc[0, 0]

    total_mols = vault_mols
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, "r") as f:
            state_data = json.load(f)
            total_mols = max(vault_mols, state_data.get("total_evaluated", vault_mols))

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Candidates Evaluated", f"{total_mols:,}")
    c2.metric("Highest Unified Score", f"{best_score:.3f}" if best_score else "0.000")
    c3.metric("Golden Candidates (High BBB & Potency)", f"{golden_mols:,}")

    st.divider()

    top_candidates = pd.read_sql_query("""
        SELECT generation, total_score, pred_pic50, bbb_score, sa_score, mw, logp, tpsa, smiles 
        FROM discovered_molecules 
        ORDER BY total_score DESC 
        LIMIT 12
    """, conn)
    conn.close()

    st.subheader("🏆 Top Discovered Scaffolds (2D Structure)")
    cols = st.columns(4)
    for index, row in top_candidates.iterrows():
        mol = Chem.MolFromSmiles(row["smiles"])
        if mol:
            img = Draw.MolToImage(mol, size=(300, 300))
            with cols[index % 4]:
                st.image(img, use_container_width=True)
                st.markdown(f"**Rank {index + 1}** (Gen {row['generation']})")
                st.markdown(
                    f"Score: `{row['total_score']:.2f}` | pIC50: `{row['pred_pic50']:.2f}` | BBB: `{row['bbb_score']:.2f}`"
                )

    st.divider()

    # Interactive 3D Conformer Studio
    st.subheader("🔬 3D Atomic Conformer & MMFF94s Force-Field Inspector")
    selected_rank = st.selectbox(
        "Select a Lead Candidate to inflate into 3D space:",
        options=list(range(len(top_candidates))),
        format_func=lambda i: f"Rank {i+1} | Score: {top_candidates.iloc[i]['total_score']:.4f} | SMILES: {top_candidates.iloc[i]['smiles']}"
    )

    sel_row = top_candidates.iloc[selected_rank]
    mol_block, ff_energy = generate_3d_conformer_molblock(sel_row["smiles"])

    col_3d, col_stats = st.columns([2, 1])
    with col_3d:
        if mol_block:
            view = py3Dmol.view(width=700, height=420)
            view.addModel(mol_block, "mol")
            view.setStyle({"stick": {"radius": 0.18, "colorscheme": "Jmol"}, "sphere": {"scale": 0.28, "colorscheme": "Jmol"}})
            view.setBackgroundColor("#0e1117")
            view.zoomTo()
            components.html(view._make_html(), height=440)
        else:
            st.error("Could not embed 3D coordinates for this structure.")

    with col_stats:
        st.markdown(f"### Rank {selected_rank + 1} Biophysical Profile")
        ic50_nm = 10 ** (9.0 - float(sel_row["pred_pic50"]))
        st.markdown(f"- **Estimated EGFR $IC_{{50}}$:** `{ic50_nm:.2f} nM`")
        st.markdown(f"- **Predicted $pIC_{{50}}$:** `{sel_row['pred_pic50']:.3f}`")
        st.markdown(f"- **BBB Permeability Score:** `{sel_row['bbb_score']:.3f} / 1.00`")
        st.markdown(f"- **Synthetic Accessibility (SA):** `{sel_row['sa_score']:.2f}` *(Easy < 3.5)*")
        st.markdown(f"- **Molecular Weight:** `{sel_row['mw']} Da`")
        st.markdown(f"- **Lipophilicity (LogP):** `{sel_row['logp']}`")
        st.markdown(f"- **Polar Surface Area (TPSA):** `{sel_row['tpsa']} Å²`")
        if ff_energy is not None:
            st.markdown(f"- **3D MMFF94s Force-Field Energy:** `{ff_energy} kcal/mol`")
        st.code(sel_row["smiles"], language="text")

        if mol_block:
            st.download_button(
                label=f"⬇️ Download Rank {selected_rank + 1} 3D Structure (.sdf)",
                data=mol_block + "\n$$$$\n",
                file_name=f"gbm_lead_rank_{selected_rank + 1}.sdf",
                mime="chemical/x-mdl-sdfile"
            )

    st.divider()
    st.subheader("📊 Complete Leaderboard Data")
    st.dataframe(top_candidates, use_container_width=True)
