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

# Directory setup for GitHub Actions runner
WORK_DIR = "./gbm_workspace"
os.makedirs(WORK_DIR, exist_ok=True)

DB_PATH = os.path.join(WORK_DIR, "gbm_vault.sqlite")
STATE_PATH = os.path.join(WORK_DIR, "adaptive_state.json")
CACHE_PATH = os.path.join(WORK_DIR, "egfr_training_cache.csv")

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
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
        )
        self.X_train, self.y_train = [], []
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
                    records.append({"smiles": smi, "pIC50": 9.0 - math.log10(float(val))})
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
                self.X_train.append(mol_to_fp(mol))
                self.y_train.append(float(row["pIC50"]))
        self.model.fit(np.array(self.X_train), np.array(self.y_train))

    def predict_potency(self, mol, seed_fps):
        fp_vec = mol_to_fp(mol).reshape(1, -1)
        pred_pic50 = float(self.model.predict(fp_vec)[0])
        rd_fp = MFP_GEN.GetFingerprint(mol)
        max_sim = max([DataStructs.TanimotoSimilarity(rd_fp, s_fp) for s_fp in seed_fps])

        h_acc = Descriptors.NumHAcceptors(mol)
        rings = Descriptors.RingCount(mol)
        pharm_bonus = 0.25 if (3 <= h_acc <= 7 and 2 <= rings <= 5) else -0.35
        ood_penalty = 0.0 if max_sim >= 0.18 else (0.18 - max_sim) * 4.0
        return pred_pic50 + pharm_bonus - ood_penalty

    def update_regression(self, new_fps, new_scores):
        if len(new_fps) >= 8:
            self.X_train.extend(new_fps)
            self.y_train.extend(new_scores)
            self.model.fit(np.array(self.X_train[-2500:]), np.array(self.y_train[-2500:]))

def evaluate_gbm_candidate(smiles, oracle, seed_fps):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() < 10 or mol.GetNumAtoms() > 55:
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
        "tpsa": round(tpsa, 1),
        "fp": mol_to_fp(mol)
    }

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
        "C=CC(=O)Nc1cc(Nc2nccc(-c3cn(C)c4ccccc34)n2)c(OC)cc1N(C)CCN(C)C",
        "COCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC",
        "O=NN(CCCl)C(=O)NC1CCCCC1"
    ]
    seed_fps = [MFP_GEN.GetFingerprint(Chem.MolFromSmiles(s)) for s in seed_scaffolds]
    alphabet = [
        "[C]", "[=C]", "[#C]", "[N]", "[=N]", "[O]", "[=O]", "[F]", "[Cl]", "[S]",
        "[Branch1]", "[Branch2]", "[Ring1]", "[Ring2]", "[=Branch1]", "[=Ring1]"
    ]

    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, "r") as f:
            state = json.load(f)
    else:
        state = {
            "generation": 0, "sigma": 0.22, "best_score": 0.0,
            "stagnant_generations": 0, "elite_pool": seed_scaffolds
        }

    seen_smiles = set(r[0] for r in conn.execute("SELECT smiles FROM discovered_molecules").fetchall())
    
    print(f"[GITHUB RUN] Starting 5.25-hour marathon from Generation {state['generation'] + 1}...")
    
    start_time = time.time()
    max_run_seconds = 5.25 * 3600  # 5 hours and 15 minutes

    while (time.time() - start_time) < max_run_seconds:

     while state["generation"] < target_generation:
        state["generation"] += 1
        gen, sigma = state["generation"], state["sigma"]
        candidates, new_fps, new_scores = [], [], []

        attempts = 0
        while len(candidates) < 50 and attempts < 200:
            attempts += 1
            mutant = inject_adaptive_randomness(random.choice(state["elite_pool"]), sigma, alphabet)
            if not mutant or mutant in seen_smiles:
                continue
            seen_smiles.add(mutant)
            res = evaluate_gbm_candidate(mutant, oracle, seed_fps)
            if res:
                candidates.append(res)
                new_fps.append(res["fp"])
                new_scores.append(res["pred_pic50"] + 0.5 * res["bbb_score"] - 0.1 * res["sa_score"])

        if candidates:
            candidates.sort(key=lambda x: x["total_score"], reverse=True)
            best_gen = candidates[0]

            for c in candidates[:15]:
                conn.execute("""
                    INSERT OR IGNORE INTO discovered_molecules
                    (smiles, generation, total_score, pred_pic50, bbb_score, sa_score, qed, mw, logp, tpsa, sigma_used)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (c["smiles"], gen, c["total_score"], c["pred_pic50"], c["bbb_score"],
                      c["sa_score"], c["qed"], c["mw"], c["logp"], c["tpsa"], round(sigma, 4)))
            conn.commit()

            if best_gen["total_score"] > state["best_score"]:
                state["best_score"] = best_gen["total_score"]
                state["stagnant_generations"] = 0
                state["sigma"] = max(0.06, sigma * 0.88)
            else:
                state["stagnant_generations"] += 1
                if state["stagnant_generations"] >= 4:
                    state["sigma"] = min(0.48, sigma * 1.25)
                    state["stagnant_generations"] = 0

            elites = [r[0] for r in conn.execute(
                "SELECT smiles FROM discovered_molecules WHERE bbb_score >= 0.65 AND sa_score <= 4.2 ORDER BY total_score DESC LIMIT 12"
            ).fetchall()]
            if elites:
                state["elite_pool"] = elites

            oracle.update_regression(new_fps, new_scores)

        if gen % 10 == 0:
            print(f"[Gen {gen:04d} | σ={sigma:.3f}] Best Score: {state['best_score']:.3f}")

    conn.close()
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)
    print(f"[GITHUB RUN] Batch complete. Reached Generation {state['generation']}. Saving vault.")

if __name__ == "__main__":
    main()
