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
from rdkit.Chem import AllChem, Descriptors, QED, DataStructs, rdFingerprintGenerator
from rdkit.Contrib.SA_Score import sascorer

RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings("ignore")

WORK_DIR = "./gbm_workspace"
os.makedirs(WORK_DIR, exist_ok=True)

DB_PATH = os.path.join(WORK_DIR, "gbm_vault.sqlite")
STATE_PATH = os.path.join(WORK_DIR, "adaptive_state.json")
CACHE_PATH = os.path.join(WORK_DIR, "egfr_training_cache.csv")

MAX_VAULT_SIZE = 25000
MFP_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

FORBIDDEN_SMARTS = [
    Chem.MolFromSmarts("[S,s]#[*,#1]"),
    Chem.MolFromSmarts("[CX3]=[SX1]"),
    Chem.MolFromSmarts("[OX2][OX2]"),
    Chem.MolFromSmarts("[NX3,NX2][F,Cl,Br]"),
    Chem.MolFromSmarts("[OX2][F,Cl,Br]"),
    Chem.MolFromSmarts("[C,c]=[C,c]=[C,c]"),
]

def mol_to_fp(mol):
    fp = MFP_GEN.GetFingerprint(mol)
    arr = np.zeros((2048,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def passes_medicinal_chemistry_filters(mol):
    if mol is None:
        return False
    num_atoms = mol.GetNumAtoms()
    if num_atoms < 20 or num_atoms > 42:
        return False
    for pattern in FORBIDDEN_SMARTS:
        if pattern is not None and mol.HasSubstructMatch(pattern):
            return False
    return True

def evaluate_3d_conformer_physics(mol):
    """
    Inflates the 2D molecule into 3D atomic coordinates (ETKDGv3) and calculates
    MMFF94s force-field energy per atom to reward low-strain, pocket-ready 3D geometries.
    """
    try:
        mol_3d = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        if AllChem.EmbedMolecule(mol_3d, params) == -1:
            return -0.50  # Penalty if molecule is sterically impossible to fold in 3D

        props = AllChem.MMFFGetMoleculeProperties(mol_3d, mmffVariant="MMFF94s")
        if props is None:
            return 0.0
        ff = AllChem.MMFFGetMoleculeForceField(mol_3d, props)
        if ff is None:
            return 0.0
        ff.Minimize(maxIts=150)
        energy_per_atom = ff.CalcEnergy() / mol.GetNumAtoms()

        # Low strain energy (< 2.5 kcal/mol per heavy atom) gets a 3D stability bonus
        if energy_per_atom < 1.8:
            return 0.45
        elif energy_per_atom < 3.5:
            return 0.20
        elif energy_per_atom > 6.5:
            return -0.60
        return 0.0
    except Exception:
        return 0.0

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
            "target_chembl_id=CHEMBL203&standard_type=IC50"
            "&standard_units=nM&standard_relation=%3D&limit=1000"
        )
        records = []
        try:
            resp = requests.get(url, timeout=15)
            for act in resp.json().get("activities", []):
                smi = act.get("canonical_smiles")
                val = act.get("standard_value")
                if smi and val and float(val) > 0:
                    pic50 = 9.0 - math.log10(float(val))
                    if 3.0 <= pic50 <= 10.5:
                        records.append({"smiles": smi, "pIC50": pic50})
        except Exception:
            pass

        curated_gbm_seeds = [
            ("Cn1nnc2c(C(N)=O)ncn2c1=O", 5.20),
            ("COCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC", 8.15),
            ("COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1", 7.85),
            ("C=CC(=O)Nc1cc(Nc2nccc(-c3cn(C)c4ccccc34)n2)c(OC)cc1N(C)CCN(C)C", 8.70),
            ("Nc1nc(OCC2CCCCC2)c2nc[nH]c2n1", 6.80),
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
        self.model.fit(
            np.array(self.X_ground_truth),
            np.array(self.y_ground_truth)
        )

    def predict_potency(self, mol, seed_fps):
        fp_vec = mol_to_fp(mol).reshape(1, -1)
        raw_pred = float(self.model.predict(fp_vec)[0])

        rd_fp = MFP_GEN.GetFingerprint(mol)
        sims = DataStructs.BulkTanimotoSimilarity(rd_fp, self.gt_rdkit_fps)
        max_sim = max(sims) if sims else 0.0

        h_acc = Descriptors.NumHAcceptors(mol)
        rings = Descriptors.RingCount(mol)
        arom = Descriptors.NumAromaticRings(mol)
        cond = (3 <= h_acc <= 7 and 3 <= rings <= 5 and arom >= 2)
        pharm_bonus = 0.25 if cond else -0.45

        ood_penalty = 0.0 if max_sim >= 0.28 else (0.28 - max_sim) * 6.5
        realistic = max(3.0, min(9.8, raw_pred + pharm_bonus - ood_penalty))
        return realistic

def evaluate_gbm_candidate(smiles, oracle, seed_fps):
    mol = Chem.MolFromSmiles(smiles)
    if not passes_medicinal_chemistry_filters(mol):
        return None

    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd = Descriptors.NumHDonors(mol)

    bbb_mw = max(0.0, min(1.0, (480.0 - mw) / 140.0))
    bbb_logp = max(0.0, 1.0 - abs(logp - 2.8) / 2.2)
    bbb_tpsa = max(0.0, min(1.0, (100.0 - tpsa) / 60.0))
    bbb_hbd = max(0.0, min(1.0, (3.0 - hbd) / 2.0))
    bbb_score = (
        0.30 * bbb_tpsa + 0.30 * bbb_logp +
        0.25 * bbb_mw + 0.15 * bbb_hbd
    )

    try:
        sa_score = sascorer.calculateScore(mol)
    except Exception:
        sa_score = 6.0

    if sa_score > 4.2:
        return None

    qed_score = QED.qed(mol)
    herg_risk = max(0.0, (logp - 4.0) * 0.8) if logp > 4.0 else 0.0
    pred_pic50 = oracle.predict_potency(mol, seed_fps)

    # Run 3D MMFF94s force-field physics check on promising candidates
    bonus_3d = 0.0
    if pred_pic50 >= 7.2 and bbb_score >= 0.65:
        bonus_3d = evaluate_3d_conformer_physics(mol)

    sa_penalty = max(0.0, (sa_score - 3.0) * 0.50)
    total_score = (
        (1.35 * pred_pic50) + (3.00 * bbb_score) +
        (1.50 * qed_score) + bonus_3d - sa_penalty - herg_risk
    )

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

def select_diverse_elites(conn, max_elites=12, sim_threshold=0.65):
    """
    Prevents the AI from getting stuck mutating clones of a single molecule.
    Selects up to 12 top-scoring scaffolds that are structurally distinct from each other.
    """
    q = (
        "SELECT smiles FROM discovered_molecules "
        "WHERE bbb_score >= 0.72 AND sa_score <= 3.6 "
        "ORDER BY total_score DESC LIMIT 250"
    )
    rows = conn.execute(q).fetchall()
    diverse_smiles = []
    diverse_fps = []

    for (smi,) in rows:
        mol = Chem.MolFromSmiles(smi)
        if not mol:
            continue
        fp = MFP_GEN.GetFingerprint(mol)
        if not diverse_fps:
            diverse_smiles.append(smi)
            diverse_fps.append(fp)
        else:
            sims = DataStructs.BulkTanimotoSimilarity(fp, diverse_fps)
            if max(sims) < sim_threshold:
                diverse_smiles.append(smi)
                diverse_fps.append(fp)
        if len(diverse_smiles) >= max_elites:
            break

    return diverse_smiles

def clean_and_recalibrate_vault(conn, oracle, seed_fps):
    q_max = "SELECT MAX(pred_pic50) FROM discovered_molecules"
    max_pic50 = conn.execute(q_max).fetchone()[0]
    if max_pic50 is not None and max_pic50 > 9.85:
        conn.execute("DELETE FROM discovered_molecules WHERE pred_pic50 > 9.85")
        conn.commit()

    q_cnt = "SELECT COUNT(*) FROM discovered_molecules"
    total_rows = conn.execute(q_cnt).fetchone()[0]
    if total_rows > MAX_VAULT_SIZE:
        conn.execute(f"""
            DELETE FROM discovered_molecules 
            WHERE smiles NOT IN (
                SELECT smiles FROM discovered_molecules 
                ORDER BY total_score DESC LIMIT {MAX_VAULT_SIZE}
            )
        """)
        conn.commit()

    conn.execute("VACUUM")
    conn.commit()

def inject_adaptive_randomness(parent_smiles, sigma, alphabet):
    try:
        tokens = list(sf.split_selfies(sf.encoder(parent_smiles)))
    except Exception:
        return None
    if not tokens:
        return None

    num_mutations = max(1, int(abs(random.gauss(0, sigma * 6)) + 1))
    for _ in range(num_mutations):
        r = random.random()
        if r < (0.35 * sigma) and len(tokens) < 46:
            idx = random.randint(0, len(tokens))
            tokens.insert(idx, random.choice(alphabet))
        elif r < (0.50 * sigma) and len(tokens) > 22:
            tokens.pop(random.randint(0, len(tokens) - 1))
        else:
            idx = random.randint(0, len(tokens) - 1)
            tokens[idx] = random.choice(alphabet)

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
        "COCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC",
        "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
        "C=CC(=O)Nc1cc(Nc2nccc(-c3cn(C)c4ccccc34)n2)c(OC)cc1N(C)CCN(C)C",
        "CN1CCN(Cc2ccc(NC(=O)c3ccc(C)c(Nc4nccc(-c5cccnc5)n4)c3)cc2)CC1"
    ]
    seed_fps = [
        MFP_GEN.GetFingerprint(Chem.MolFromSmiles(s))
        for s in seed_scaffolds
    ]
    alphabet = [
        "[C]", "[=C]", "[#C]", "[N]", "[=N]", "[O]", "[=O]",
        "[F]", "[Cl]", "[Branch1]", "[Branch2]",
        "[Ring1]", "[Ring2]", "[=Branch1]", "[=Ring1]"
    ]

    q_cnt = "SELECT COUNT(*) FROM discovered_molecules"
    initial_db_count = conn.execute(q_cnt).fetchone()[0]

    clean_and_recalibrate_vault(conn, oracle, seed_fps)

    q_best = "SELECT MAX(total_score) FROM discovered_molecules"
    true_best = conn.execute(q_best).fetchone()[0] or 0.0

    elites = select_diverse_elites(conn)

    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, "r") as f:
            state = json.load(f)
        state["best_score"] = float(true_best)
        prev_eval = state.get("total_evaluated", 0)
        state["total_evaluated"] = max(prev_eval, initial_db_count)
        state["elite_pool"] = elites if elites else seed_scaffolds
    else:
        state = {
            "generation": 0,
            "sigma": 0.22,
            "best_score": float(true_best),
            "total_evaluated": initial_db_count,
            "stagnant_generations": 0,
            "elite_pool": elites if elites else seed_scaffolds
        }

    q_all = "SELECT smiles FROM discovered_molecules"
    seen_smiles = set(r[0] for r in conn.execute(q_all).fetchall())
    print(f"[AUTOPILOT] Starting 3D-Aware Marathon | Best: {state['best_score']:.3f}")

    start_time = time.time()
    max_run_seconds = int(4.5 * 3600)

    while (time.time() - start_time) < max_run_seconds:
        state["generation"] += 1
        gen, sigma = state["generation"], state["sigma"]
        candidates = []

        attempts = 0
        while len(candidates) < 50 and attempts < 220:
            attempts += 1
            pool = state["elite_pool"] if random.random() < 0.75 else seed_scaffolds
            parent = random.choice(pool)
            mutant = inject_adaptive_randomness(parent, sigma, alphabet)
            if not mutant or mutant in seen_smiles:
                continue
            seen_smiles.add(mutant)
            res = evaluate_gbm_candidate(mutant, oracle, seed_fps)
            if res:
                candidates.append(res)
                state["total_evaluated"] += 1

        if candidates:
            candidates.sort(key=lambda x: x["total_score"], reverse=True)
            best_gen = candidates[0]

            for c in candidates[:10]:
                conn.execute("""
                    INSERT OR IGNORE INTO discovered_molecules
                    (smiles, generation, total_score, pred_pic50,
                     bbb_score, sa_score, qed, mw, logp, tpsa, sigma_used)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    c["smiles"], gen, c["total_score"], c["pred_pic50"],
                    c["bbb_score"], c["sa_score"], c["qed"], c["mw"],
                    c["logp"], c["tpsa"], round(sigma, 4)
                ))
            conn.commit()

            if best_gen["total_score"] > state["best_score"]:
                state["best_score"] = best_gen["total_score"]
                state["stagnant_generations"] = 0
                state["sigma"] = max(0.06, sigma * 0.88)
            else:
                state["stagnant_generations"] += 1
                if state["stagnant_generations"] >= 4:
                    state["sigma"] = min(0.45, sigma * 1.25)
                    state["stagnant_generations"] = 0

            if gen % 5 == 0:
                elites = select_diverse_elites(conn)
                if elites:
                    state["elite_pool"] = elites

        if gen % 25 == 0 and candidates:
            print(f"[Gen {gen:05d}] Best: {state['best_score']:.3f} | Pool: {len(state['elite_pool'])}")

    clean_and_recalibrate_vault(conn, oracle, seed_fps)
    conn.close()

    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)
    print("[AUTOPILOT] 4.5-Hour Marathon Complete. Vault saved.")

if __name__ == "__main__":
    main()
