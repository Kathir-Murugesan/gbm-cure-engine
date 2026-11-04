import os
import json
import sqlite3
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import py3Dmol
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

st.set_page_config(page_title="GBM AI Dashboard", layout="wide", page_icon="🧬")

st.title("🧬 Glioblastoma De Novo Drug Discovery")
st.caption("Live telemetry, 3D EGFR (PDB: 1M17) crystal pocket docking, and lead export.")

DB_PATH = "gbm_workspace/gbm_vault.sqlite"
STATE_PATH = "gbm_workspace/adaptive_state.json"
SDF_PATH = "gbm_workspace/top_25_docked_leads.sdf"
PDB_PATH = "gbm_workspace/1M17.pdb"

def load_docked_sdf_leads():
    leads = {}
    if os.path.exists(SDF_PATH):
        suppl = Chem.SDMolSupplier(SDF_PATH, removeHs=False)
        for mol in suppl:
            if mol is not None and mol.HasProp("SMILES"):
                smi = mol.GetProp("SMILES")
                dg = mol.GetProp("Estimated_3D_DeltaG_kcal_mol") if mol.HasProp("Estimated_3D_DeltaG_kcal_mol") else "N/A"
                leads[smi] = (Chem.MolToMolBlock(mol), dg)
    return leads

def extract_pocket_pdb_block(center_coords=None, radius=11.0):
    if not os.path.exists(PDB_PATH):
        return None
    pocket_lines = []
    with open(PDB_PATH, "r") as f:
        for line in f:
            if line.startswith("ATOM"):
                if center_coords is not None:
                    try:
                        x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                        if np.linalg.norm(np.array([x, y, z]) - center_coords) <= radius:
                            pocket_lines.append(line)
                    except Exception:
                        pass
                else:
                    pocket_lines.append(line)
    return "".join(pocket_lines) + "END\n" if pocket_lines else None

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

    st.subheader("🔬 3D EGFR Active Pocket (PDB: 1M17) Docking Studio")
    docked_leads = load_docked_sdf_leads()

    selected_rank = st.selectbox(
        "Select a Lead Candidate to inspect inside the 3D EGFR Kinase Pocket:",
        options=list(range(len(top_candidates))),
        format_func=lambda i: f"Rank {i+1} | Score: {top_candidates.iloc[i]['total_score']:.4f} | SMILES: {top_candidates.iloc[i]['smiles']}"
    )

    sel_row = top_candidates.iloc[selected_rank]
    smi = sel_row["smiles"]

    mol_block, est_dg = docked_leads.get(smi, (None, "Pending next SDF export"))
    pocket_pdb = None

    if mol_block:
        docked_mol = Chem.MolFromMolBlock(mol_block)
        if docked_mol and docked_mol.GetNumConformers() > 0:
            conf = docked_mol.GetConformer()
            pts = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
                            for i in range(docked_mol.GetNumAtoms())])
            pocket_pdb = extract_pocket_pdb_block(np.mean(pts, axis=0), radius=10.5)
    else:
        mol = Chem.AddHs(Chem.MolFromSmiles(smi))
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        AllChem.MMFFOptimizeMolecule(mol, mmffVariant="MMFF94s")
        mol_block = Chem.MolToMolBlock(mol)

    col_3d, col_stats = st.columns([2, 1])
    with col_3d:
        show_pocket = st.checkbox("Show Surrounding 3D EGFR Protein Pocket (PDB: 1M17)", value=True)
        view = py3Dmol.view(width=700, height=450)
        if show_pocket and pocket_pdb:
            view.addModel(pocket_pdb, "pdb")
            view.setStyle({"model": 0}, {"line": {"colorscheme": "whiteCarbon", "opacity": 0.45}})
            view.addSurface(py3Dmol.VDW, {"opacity": 0.22, "color": "#3b82f6"}, {"model": 0})
            view.addModel(mol_block, "mol")
            view.setStyle({"model": 1}, {"stick": {"radius": 0.22, "colorscheme": "greenCarbon"}, "sphere": {"scale": 0.28}})
            view.zoomTo({"model": 1})
        else:
            view.addModel(mol_block, "mol")
            view.setStyle({"stick": {"radius": 0.20, "colorscheme": "Jmol"}, "sphere": {"scale": 0.28, "colorscheme": "Jmol"}})
            view.zoomTo()
        view.setBackgroundColor("#0e1117")
        components.html(view._make_html(), height=460)

    with col_stats:
        st.markdown(f"### Rank {selected_rank + 1} Biophysical Profile")
        ic50_nm = 10 ** (9.0 - float(sel_row["pred_pic50"]))
        st.markdown(f"- **3D EGFR Docking $\Delta G$:** `{est_dg} kcal/mol`")
        st.markdown(f"- **Estimated EGFR $IC_{{50}}$:** `{ic50_nm:.2f} nM`")
        st.markdown(f"- **Predicted $pIC_{{50}}$:** `{sel_row['pred_pic50']:.3f}`")
        st.markdown(f"- **BBB Permeability Score:** `{sel_row['bbb_score']:.3f} / 1.00`")
        st.markdown(f"- **Synthetic Accessibility (SA):** `{sel_row['sa_score']:.2f}`")
        st.markdown(f"- **Molecular Weight:** `{sel_row['mw']} Da`")
        st.markdown(f"- **Lipophilicity (LogP):** `{sel_row['logp']}`")
        st.markdown(f"- **Polar Surface Area (TPSA):** `{sel_row['tpsa']} Å²`")
        st.code(smi, language="text")

        if os.path.exists(SDF_PATH):
            with open(SDF_PATH, "rb") as f:
                st.download_button(
                    label="⬇️ Download All Top 25 3D Docked Leads (.sdf)",
                    data=f.read(),
                    file_name="top_25_docked_leads.sdf",
                    mime="chemical/x-mdl-sdfile"
                )

    st.divider()
    st.subheader("📊 Complete Leaderboard Data")
    st.dataframe(top_candidates, use_container_width=True)
