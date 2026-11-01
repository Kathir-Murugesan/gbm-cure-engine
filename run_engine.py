import os
import time
import json
import math
import random
import sqlite3
import warnings
import requests
import numpy as np
import pandas as pd
import selfies as sf
import xgboost as xgb
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, QED, DataStructs, rdFingerprintGenerator
from rdkit.Contrib.SA_Score import sascorer

RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings("ignore")

WORK_DIR = "./gbm_workspace"
os.makedirs(WORK_DIR, exist_ok=True)

DB_PATH = os.path.join(WORK_DIR, "gbm_vault.sqlite")
STATE_PATH = os.path.join(WORK_DIR, "adaptive_state.json")
CACHE_PATH = os.path.join(WORK_DIR, "egfr_training_cache.csv")

# Hard limit to prevent GitHub's 100MB file limit crash
MAX_VAULT_SIZE = 25000

MFP_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def mol_to_fp(mol):
    fp = MFP_GEN.GetFingerprint(mol)
    arr = np.zeros((2048,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

class GBMRegressionOracle:
    def __init__(self, cache_path):
        self.cache_path = cache_path
        self.model = xgb.XGBRegressor(
            n_estimators=250, max_depth=6, learning_rate=0.04,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
        )
        self.X_ground_truth = []
        self.y_ground_truth = []
        self.gt_rdkit_fps = []
        self._initialize_brain()

    def _fetch_chembl_egfr_data(self):
        if os.path.exists(self.cache_path):
            return pd.read_csv(self.cache_path)

        url = (
            "https://www.ebi.ac.uk/chembl/api/data/activity.json?"
            "target_chembl_id=CHEMBL203&standard_type=IC50&standard_units=nM"
            "&standard_relation=%3D&limit=1000"
        )
        records = []
        try:
            resp = requests.get(url, timeout=15)
            for act in resp.json().get("activities", []):
                smi, val = act.get("canonical_smiles"), act.get("standard_value")
                if smi and val and float(val) > 0:
                    pic50 = 9.0 - math.log10(float(val))
                    if 3.0 <= pic50 <= 11.5:
                        records.append({"smiles": smi, "pIC50": pic50})
        except Exception:
            pass

        curated_gbm_seeds = [
            ("Cn1nnc2c(C(N)=O)ncn2c1=O", 5.20),
            ("COCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC", 8.15),
            ("COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1", 7.85),
            ("C=CC(=O)Nc1cc(Nc2nccc(-c3cn(C)c4ccccc34)n2)c(OC)cc1N(C)CCN(C)C", 8.70),
            ("CS(=O)(=O)CCNCc1ccc(-c2ccc3ncnc(Nc4ccc(OCc5cccc(F)c5)c(Cl)c4)c3c2)o1", 7.92),
            ("Nc1nc(OCC2CCCCC2)c2nc[nH]c2n1", 6.80),
            ("CN1CCN(Cc2ccc(NC(=O)c3ccc(C)c(Nc4nccc(-c5cccnc5)n4)c3)cc2)CC1", 6.40),
            ("c1ccc2c(c1)ccc1ccccc12", 4.10),
            ("CC(=O)Oc1ccccc1C(=O)O", 3.50),
            ("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", 3.80)
        ]
        for smi, pic50 in curated_gbm_seeds:
            records.append({"smiles": smi, "pIC50": pic50})

        df = pd.DataFrame(records).drop_duplicates(subset=["smiles"])
        df.to_csv(self.cache_path, index=False)
        return df

    def _initialize_brain(self):
        df = self._fetch_chembl_egfr_data()
        for _, row in df.iterrows():
            mol = Chem.MolFromSmiles(row["smiles"])
            if mol:
                self.X_ground_truth.append(mol_to_fp(mol))
                self.y_ground_truth.append(float(row["pIC50"]))
                self.gt_rdkit_fps.append(MFP_GEN.GetFingerprint(mol))
        self.model.fit(np.array(self.X_ground_truth), np.array(self.y_ground_truth))

    def predict_potency(self, mol, seed_fps):
        fp_vec = mol_to_fp(mol).reshape(1, -1)
        raw_pred = float(self.model.predict(fp_vec)[0])

        rd_fp = MFP_GEN.GetFingerprint(mol)
        sims = DataStructs.BulkTanimotoSimilarity(rd_fp, self.gt_rdkit_fps)
        max_sim = max(sims) if sims else 0.0

        h_acc = Descriptors.NumHAcceptors(mol)
        rings = Descriptors.RingCount(mol)
        arom_rings = Descriptors.NumAromaticRings(mol)
        pharm_bonus = 0.25 if (3 <= h_acc <= 7 and 2 <= rings <= 5 and arom_rings >= 2) else -0.40

        ood_penalty = 0.0 if max_sim >= 0.25 else (0.25 - max_sim) * 6.0
        # HARD PHYSICS CAP: Cannot exceed 10.8
        realistic_pic50 = max(3.0, min(10.8, raw_pred + pharm_bonus - ood_penalty))
        return realistic_pic50

def evaluate_gbm_candidate(smiles, oracle, seed_fps):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() < 12 or mol.GetNumAtoms() > 50:
        return None

    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd = Descriptors.NumHDonors(mol)

    bbb_mw = max(0.0, min(1.0, (500.0 - mw) / 150.0))
    bbb_logp = max(0.0, 1.0 - abs(logp - 2.75) / 2.5)
    bbb_tpsa = max(0.0, min(1.0, (105.0 - tpsa) / 65.0))
    bbb_hbd = max(0.0, min(1.0, (3.0 - hbd) / 2.0))
    bbb_score = 0.30 * bbb_tpsa + 0.30 * bbb_logp + 0.25 * bbb_mw + 0.15 * bbb_hbd

    try:
        sa_score = sascorer.calculateScore(mol)
    except Exception:
        sa_score = 6.0

    qed_score = QED.qed(mol)
    herg_risk = max(0.0, (logp - 4.2) * 0.8) if logp > 4.2 else 0.0
    pred_pic50 = oracle.predict_potency(mol, seed_fps)

    sa_penalty = max(0.0, (sa_score - 3.2) * 0.45)
    total_score = (1.35 * pred_pic50) + (3.00 * bbb_score) + (1.50 * qed_score) - sa_penalty - herg_risk

    return {
        "smiles": Chem.MolToSmiles(mol, canonical=True),
        "total_score": round(total_score, 4),
        "pred_pic50": round(pred_pic50, 3),
        "bbb_score": round(bbb_score, 3),
        "sa_score": round(sa_score, 2),
        "qed": round(qed_score, 3),
        "mw": round(mw, 1),
        "logp": round(logp, 2),
        "tpsa": round(tpsa, 1)
    }

def instant_rescue_and_prune(conn):
    """ Instantly deletes hallucinated scores and prunes the DB to prevent GitHub crashes. """
    max_pic50 = conn.execute("SELECT MAX(pred_pic50) FROM discovered_molecules").fetchone()[0]
    if max_pic50 is not None and max_pic50 > 11.0:
        print(f"[AUDIT] Detected impossible physics (pIC50={max_pic50:.2f}). Nuking corrupted data...")
        conn.execute("DELETE FROM discovered_molecules WHERE pred_pic50 > 11.0")
        conn.commit()
        print("[AUDIT] Corrupted molecules deleted instantly.")

    total_rows = conn.execute("SELECT COUNT(*) FROM discovered_molecules").fetchone()[0]
    if total_rows > MAX_VAULT_SIZE:
        print(f"[PRUNE] Trimming vault to top {MAX_VAULT_SIZE} to ensure small file size...")
        conn.execute(f"""
            DELETE FROM discovered_molecules 
            WHERE smiles NOT IN (
                SELECT smiles FROM discovered_molecules 
                ORDER BY total_score DESC LIMIT {MAX_VAULT_SIZE}
            )
        """)
        conn.commit()
    
    print("[VAULT] Running compression (VACUUM)...")
    conn.execute("VACUUM")
    conn.commit()

def inject_adaptive_randomness(parent_smiles, sigma, alphabet):
    try:
        tokens = list(sf.split_selfies(sf.encoder(parent_smiles)))
    except Exception:
        return None
    if not tokens:
        return None

    num_mutations = max(1, int(abs(random.gauss(0, sigma * 8)) + 1))
    for _ in range(num_mutations):
        r = random.random()
        if r < (0.35 * sigma) and len(tokens) < 48:
            tokens.insert(random.randint(0, len(tokens)), random.choice(alphabet))
        elif r < (0.55 * sigma) and len(tokens) > 12:
            tokens.pop(random.randint(0, len(tokens) - 1))
        else:
            tokens[random.randint(0, len(tokens) - 1)] = random.choice(alphabet)

    try:
        mol = Chem.MolFromSmiles(sf.decoder("".join(tokens)))
        if mol:
            return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        pass
    return None

def main():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS discovered_molecules (
            smiles TEXT PRIMARY KEY,
            generation INT,
            total_score REAL,
            pred_pic50 REAL,
            bbb_score REAL,
            sa_score REAL,
            qed REAL,
            mw REAL,
            logp REAL,
            tpsa REAL,
            sigma_used REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()

    oracle = GBMRegressionOracle(CACHE_PATH)
    
    seed_scaffolds = [
        "Cn1nnc2c(C(N)=O)ncn2c1=O",
        "COCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC",
        "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
        "C=CC(=O)Nc1cc(Nc2nccc(-c3cn(C)c4ccccc34)n2)c(OC)cc1N(C)CCN(C)C"
    ]
    seed_fps = [MFP_GEN.GetFingerprint(Chem.MolFromSmiles(s)) for s in seed_scaffolds]
    alphabet = ["[C]", "[=C]", "[#C]", "[N]", "[=N]", "[O]", "[=O]", "[F]", "[Cl]", "[S]", "[Branch1]", "[Branch2]", "[Ring1]", "[Ring2]", "[=Branch1]", "[=Ring1]"]

    initial_db_count = conn.execute("SELECT COUNT(*) FROM discovered_molecules").fetchone()[0]
    
    # Run instant cleanup
    instant_rescue_and_prune(conn)

    true_best = conn.execute("SELECT MAX(total_score) FROM discovered
