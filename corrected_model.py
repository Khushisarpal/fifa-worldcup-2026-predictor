"""
FIFA World Cup 2026 — Prediction Model
Predicts per-team probability of: reaching Semis, Final, and Winning
Uses: Elo (time-decay + weighted k) + Poisson + XGBoost + Monte Carlo
"""

import os, warnings, itertools
import numpy as np
import pandas as pd
from scipy.stats import poisson
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
os.makedirs("output", exist_ok=True)


# CONFIG

KAGGLE_CSV    = "data/results.csv"
LIVE_XLSX     = "data/dataa live.xlsx"
LIVE_SHEET    = "data live"
N_SIMULATIONS = int(os.environ.get("N_SIM", "10000"))
REFERENCE_DATE = pd.Timestamp("2026-06-28")   # day after group stage ends

# Team name mapping: live/fixture names → historical names
NAME_MAP = {
    "Korea Republic":       "South Korea",
    "Czechia":              "Czech Republic",
    "USA":                  "United States",
    "Trkiye":               "Turkey",
    "Cte dIvoire":          "Ivory Coast",
    "Curao":                "Curaçao",
    "Cabo Verde":           "Cape Verde",
    "Côte d'Ivoire":        "Ivory Coast",
    "Curaçao":              "Curaçao",
}

def norm(name):
    """Normalize team name to historical standard."""
    return NAME_MAP.get(str(name).strip(), str(name).strip())


# PHASE 1 — LOAD DATA

print("=" * 60)
print("PHASE 1 — Loading data")
print("=" * 60)

# Historical
hist = pd.read_csv(KAGGLE_CSV)
hist.columns = [c.strip().lower() for c in hist.columns]
hist["date"] = pd.to_datetime(hist["date"], format="mixed", dayfirst=False, errors="coerce")
hist = hist.dropna(subset=["date"])
hist = hist[hist["home_score"].notna() & hist["away_score"].notna()].copy()
hist["home_score"] = hist["home_score"].astype(int)
hist["away_score"] = hist["away_score"].astype(int)
hist["neutral"]    = hist["neutral"].astype(str).str.upper().str.strip() == "TRUE"
hist["home_team"]  = hist["home_team"].apply(norm)
hist["away_team"]  = hist["away_team"].apply(norm)
hist = hist[["date","home_team","away_team","home_score","away_score","tournament","neutral"]]

# Live WC 2026 results
live = pd.read_excel(LIVE_XLSX, sheet_name=LIVE_SHEET)
live.columns = [c.strip().lower() for c in live.columns]
live["date"]       = pd.to_datetime(live["date"], errors="coerce")
live               = live.dropna(subset=["date","home_score","away_score"])
live["home_score"] = live["home_score"].astype(int)
live["away_score"] = live["away_score"].astype(int)
live["neutral"]    = live["neutral"].astype(str).str.upper().str.strip().isin(["TRUE","1","YES"]) | (live["neutral"] == True)
live["home_team"]  = live["home_team"].apply(norm)
live["away_team"]  = live["away_team"].apply(norm)
live = live[["date","home_team","away_team","home_score","away_score","tournament","neutral"]]

# Combine — live results override historical for same fixture
df = (pd.concat([hist, live], ignore_index=True)
        .sort_values("date")
        .drop_duplicates(subset=["date","home_team","away_team"], keep="last")
        .reset_index(drop=True))

print(f"  ✓ Historical : {len(hist):,} matches")
print(f"  ✓ Live WC    : {len(live):,} matches (already played)")
print(f"  ✓ Combined   : {len(df):,} matches | {df['date'].min().date()} → {df['date'].max().date()}")


# PHASE 2 — ELO RATINGS (time-decay + tournament-weighted k)

print("\n" + "=" * 60)
print("PHASE 2 — Elo ratings")
print("=" * 60)

HALF_LIFE   = 365 * 3   # 3-year half-life
HOME_ADV    = 65.0
BASE_K      = 32.0

T_WEIGHT = {
    "FIFA World Cup": 1.00, "Confederations Cup": 0.85,
    "Copa América": 0.85, "UEFA Euro": 0.85,
    "African Cup of Nations": 0.75, "Africa Cup of Nations": 0.75,
    "AFC Asian Cup": 0.75, "CONCACAF Gold Cup": 0.70,
    "CONCACAF Nations League": 0.65, "UEFA Nations League": 0.65,
    "FIFA World Cup qualification": 0.60,
    "UEFA Euro qualification": 0.55,
    "Friendly": 0.30,
}

def tw(name):
    if name in T_WEIGHT: return T_WEIGHT[name]
    for k, v in T_WEIGHT.items():
        if k.lower() in name.lower(): return v
    return 0.45

def tdecay(d):
    days = max((REFERENCE_DATE - d).days, 0)
    return 2 ** (-days / HALF_LIFE)

def kfactor(d, tourn, hg, ag):
    margin = 1 + np.log1p(abs(hg - ag)) * 0.5
    return float(np.clip(BASE_K * tw(tourn) * tdecay(d) * margin, 6, 64))

def exp_s(a, b): return 1 / (1 + 10 ** ((b - a) / 400))

def act_s(hg, ag):
    if hg > ag: return 1.0, 0.0
    if hg < ag: return 0.0, 1.0
    return 0.5, 0.5


# Initialize Elo dictionary and new columns in df to store historical ratings
elo = {}
df["elo_home_pre"] = 1500.0
df["elo_away_pre"] = 1500.0

print("  Processing Elo ratings in a single pass with recency weighting...")

# Define the cutoff date for the 3-year recency boost
recency_cutoff = REFERENCE_DATE - pd.DateOffset(years=3)

for idx, r in df.iterrows():
    ht, at = r["home_team"], r["away_team"]
    hg, ag = int(r["home_score"]), int(r["away_score"])
    
    # 1. Fetch current pre-match ratings
    eh = elo.get(ht, 1500.0)
    ea = elo.get(at, 1500.0)
    
    # [CRITICAL FIX]: Save current pre-match Elo directly to the dataframe.
    # This prevents your XGBoost/Logistic Regression models from cheating in Phase 3.
    df.at[idx, "elo_home_pre"] = eh
    df.at[idx, "elo_away_pre"] = ea
    
    # 2. Calculate expectations and actual outcomes
    hb = 0 if r["neutral"] else HOME_ADV
    e_h = exp_s(eh + hb, ea)
    s_h, s_a = act_s(hg, ag)
    
    # 3. Calculate baseline K-factor (handles time-decay and tournament weights)
    k = kfactor(r["date"], str(r["tournament"]), hg, ag)
    
    # 4. Apply the recency boost dynamically if the match is within the last 3 years
    if r["date"] >= recency_cutoff:
        k *= 1.3
        
    # 5. Update the master Elo ratings
    elo[ht] = eh + k * (s_h - e_h)
    elo[at] = ea + k * (s_a - (1 - e_h))

# Set the final Elo ratings for your tournament simulation cache
ELO = elo

top20 = sorted(ELO.items(), key=lambda x: -x[1])[:20]
print("  Top 20 Elo ratings:")
for i, (t, v) in enumerate(top20, 1):
    print(f"    {i:2}. {t:<28} {v:7.1f}")


# PHASE 3 — FEATURES

print("\n" + "=" * 60)
print("PHASE 3 — Feature engineering")
print("=" * 60)

def form(team, before, n=6):
    """Goals scored/conceded/points in last n matches before date."""
    mask = ((df["home_team"]==team)|(df["away_team"]==team)) & (df["date"]<before)
    played = df[mask].tail(n)
    gf=ga=pts=0
    for _,r in played.iterrows():
        if r["home_team"]==team:
            gf+=r["home_score"]; ga+=r["away_score"]
            pts+= 3 if r["home_score"]>r["away_score"] else (1 if r["home_score"]==r["away_score"] else 0)
        else:
            gf+=r["away_score"]; ga+=r["home_score"]
            pts+= 3 if r["away_score"]>r["home_score"] else (1 if r["home_score"]==r["away_score"] else 0)
    n_ = max(len(played),1)
    return gf/n_, ga/n_, pts/n_

def feats(ht, at, neutral, tourn, date, elo_h_pre=None, elo_a_pre=None):
    # If pre-calculated values are passed (for training history), use them. 
    # Otherwise fallback to current final Elo (for upcoming World Cup games).
    eh = elo_h_pre if elo_h_pre is not None else ELO.get(ht, 1500.0)
    ea = elo_a_pre if elo_a_pre is not None else ELO.get(at, 1500.0)
    
    hb = 0 if neutral else HOME_ADV
    hgf, hga, hpts = form(ht, date)
    return {
        "elo_delta":    eh - ea,
        "elo_home":     eh,
        "elo_away":     ea,
        "elo_win_prob": exp_s(eh+hb, ea),
        "form_gf_h":    hgf,  "form_ga_h": hga,  "form_pts_h": hpts,
        "form_gf_a":    hgf,  "form_ga_a": hga,  "form_pts_a": hpts,
        "neutral":      int(neutral),
        "tourn_w":      tw(tourn),
    }

FCOLS = ["elo_delta","elo_home","elo_away","elo_win_prob",
         "form_gf_h","form_ga_h","form_pts_h",
         "form_gf_a","form_ga_a","form_pts_a",
         "neutral","tourn_w"]

def outcome(r):
    if r["home_score"]>r["away_score"]: return 0
    if r["home_score"]==r["away_score"]: return 1
    return 2

print("  Building feature matrix (this takes ~3-5 min on 49k rows) …")
rows = [feats(r["home_team"],r["away_team"],bool(r["neutral"]),str(r["tournament"]),r["date"])
        for _,r in df.iterrows()]
feat = pd.DataFrame(rows)
df["outcome"] = df.apply(outcome, axis=1)
feat["outcome"] = df["outcome"].values
print(f"  Shape: {feat.shape} | W={sum(df.outcome==0):,} D={sum(df.outcome==1):,} L={sum(df.outcome==2):,}")


# PHASE 4 — POISSON MODEL (last 2 years, competitive only)

print("\n" + "=" * 60)
print("PHASE 4 — Poisson attack/defence model")
print("=" * 60)

pdf  = df[
    (df["date"] >= REFERENCE_DATE - pd.DateOffset(years=2)) &
    (df["tournament"].isin([
        "FIFA World Cup", "FIFA World Cup qualification",
        "UEFA Euro", "UEFA Euro qualification",
        "Copa América", "African Cup of Nations",
        "AFC Asian Cup", "UEFA Nations League",
        "CONCACAF Nations League", "CONCACAF Gold Cup",
    ]))
].copy()
tms  = sorted(set(pdf["home_team"]) | set(pdf["away_team"]))
tidx = {t:i for i,t in enumerate(tms)}
N    = len(tms)
print(f"  Teams: {N}  |  Matches: {len(pdf):,}")

def pnll(params):
    att=params[:N]; dfe=params[N:2*N]; hm=params[-1]
    ll=0.0
    for _,r in pdf.iterrows():
        hi=tidx.get(r["home_team"]); ai=tidx.get(r["away_team"])
        if hi is None or ai is None: continue
        hb = 0 if r["neutral"] else hm
        lh = np.exp(att[hi]-dfe[ai]+hb)
        la = np.exp(att[ai]-dfe[hi])
        ll += poisson.logpmf(int(r["home_score"]),lh) + poisson.logpmf(int(r["away_score"]),la)
    return -ll

print("  Fitting Poisson MLE …")
x0=np.zeros(2*N+1); x0[-1]=0.3
res=minimize(pnll, x0, method="L-BFGS-B",
             bounds=[(-3,3)]*(2*N)+[(0,1)],
             options={"maxiter":300,"ftol":1e-5})
ATT=dict(zip(tms,res.x[:N]))
DEF=dict(zip(tms,res.x[N:2*N]))
HP=res.x[-1]
print(f"  Converged: {res.success}  |  home_adv={HP:.3f}")

def plam(ht, at, neutral=True):
    lh = np.exp(ATT.get(ht,0.0)-DEF.get(at,0.0)+(0 if neutral else HP))
    la = np.exp(ATT.get(at,0.0)-DEF.get(ht,0.0))
    return lh, la

def pprobs(ht, at, neutral=True):
    lh,la = plam(ht,at,neutral)
    ph=pd_=pa=0.0
    for g_h in range(11):
        for g_a in range(11):
            p = poisson.pmf(g_h,lh)*poisson.pmf(g_a,la)
            if g_h>g_a: ph+=p
            elif g_h==g_a: pd_+=p
            else: pa+=p
    return ph, pd_, pa, lh, la


# PHASE 5 — XGBOOST CLASSIFIER
print("\n" + "=" * 60)
print("PHASE 5 — XGBoost classifier")
print("=" * 60)

X_raw = feat[FCOLS].values
y     = feat["outcome"].values
imp   = SimpleImputer(strategy="median")
X     = imp.fit_transform(X_raw)
scl   = StandardScaler()
Xs    = scl.fit_transform(X)

val_mask = df["date"] >= (REFERENCE_DATE - pd.DateOffset(years=2))
Xtr,Xv  = Xs[~val_mask], Xs[val_mask]
ytr,yv  = y[~val_mask],  y[val_mask]
print(f"  Train: {len(Xtr):,}  |  Val: {len(Xv):,}")

lr = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
lr.fit(Xtr, ytr)

xgb = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                     subsample=0.8, colsample_bytree=0.8,
                     eval_metric="mlogloss", verbosity=0)
xgb.fit(Xtr, ytr)
print(f"  Val accuracy (XGB): {(xgb.predict(Xv)==yv).mean():.3f}")

def cprobs(ht, at, neutral=True, date=REFERENCE_DATE):
    f  = feats(ht, at, neutral, "FIFA World Cup", date)
    x  = np.array([f[c] for c in FCOLS]).reshape(1,-1)
    x  = scl.transform(imp.transform(x))
    pl = lr.predict_proba(x)[0]
    px = xgb.predict_proba(x)[0]
    cl = list(lr.classes_)
    p  = 0.5*pl + 0.5*px
    return p[cl.index(0)], p[cl.index(1)], p[cl.index(2)]


# PHASE 6 — BLEND + CACHE ALL WC MATCHUPS

print("\n" + "=" * 60)
print("PHASE 6 — Blending & caching WC probabilities")
print("=" * 60)

# All 48 WC teams (historical names)
WC_TEAMS = [
    "Algeria","Argentina","Australia","Austria","Belgium",
    "Bosnia and Herzegovina","Brazil","Canada","Cape Verde",
    "Colombia","Croatia","Curaçao","Czech Republic","DR Congo",
    "Ecuador","Egypt","England","France","Germany","Ghana",
    "Haiti","Iran","Iraq","Ivory Coast","Japan","Jordan",
    "Mexico","Morocco","Netherlands","New Zealand","Norway",
    "Panama","Paraguay","Portugal","Qatar","Saudi Arabia",
    "Scotland","Senegal","South Africa","South Korea","Spain",
    "Sweden","Switzerland","Tunisia","Turkey","United States",
    "Uruguay","Uzbekistan"
]

# Group stage fixtures from historical (team names already normalized)
GS_FIXTURES = [
    # Group A
    ("Mexico","South Africa"), ("South Korea","Czech Republic"),
    ("Mexico","South Korea"), ("Czech Republic","South Africa"),
    ("Mexico","Czech Republic"), ("South Africa","South Korea"),
    # Group B
    ("Canada","Bosnia and Herzegovina"), ("Qatar","Switzerland"),
    ("Canada","Qatar"), ("Switzerland","Bosnia and Herzegovina"),
    ("Canada","Switzerland"), ("Bosnia and Herzegovina","Qatar"),
    # Group C
    ("Brazil","Morocco"), ("Haiti","Scotland"),
    ("Brazil","Haiti"), ("Scotland","Morocco"),
    ("Brazil","Scotland"), ("Morocco","Haiti"),
    # Group D
    ("United States","Paraguay"), ("Australia","Turkey"),
    ("United States","Australia"), ("Turkey","Paraguay"),
    ("United States","Turkey"), ("Paraguay","Australia"),
    # Group E
    ("Germany","Curaçao"), ("Ivory Coast","Ecuador"),
    ("Germany","Ivory Coast"), ("Ecuador","Curaçao"),
    ("Germany","Ecuador"), ("Curaçao","Ivory Coast"),
    # Group F
    ("Netherlands","Japan"), ("Sweden","Tunisia"),
    ("Netherlands","Sweden"), ("Tunisia","Japan"),
    ("Netherlands","Tunisia"), ("Japan","Sweden"),
    # Group G
    ("Belgium","Egypt"), ("Iran","New Zealand"),
    ("Belgium","Iran"), ("New Zealand","Egypt"),
    ("Belgium","New Zealand"), ("Egypt","Iran"),
    # Group H
    ("Spain","Cape Verde"), ("Saudi Arabia","Uruguay"),
    ("Spain","Saudi Arabia"), ("Uruguay","Cape Verde"),
    ("Spain","Uruguay"), ("Cape Verde","Saudi Arabia"),
    # Group I
    ("France","Senegal"), ("Iraq","Norway"),
    ("France","Iraq"), ("Norway","Senegal"),
    ("France","Norway"), ("Senegal","Iraq"),
    # Group J
    ("Argentina","Algeria"), ("Austria","Jordan"),
    ("Argentina","Austria"), ("Jordan","Algeria"),
    ("Argentina","Jordan"), ("Algeria","Austria"),
    # Group K
    ("Portugal","DR Congo"), ("Uzbekistan","Colombia"),
    ("Portugal","Uzbekistan"), ("Colombia","DR Congo"),
    ("Portugal","Colombia"), ("DR Congo","Uzbekistan"),
    # Group L
    ("England","Croatia"), ("Ghana","Panama"),
    ("England","Ghana"), ("Panama","Croatia"),
    ("England","Panama"), ("Croatia","Ghana"),
]

# Groups (historical team names)
GROUPS = {
   
    "A": ["Mexico",        "South Africa",          "South Korea",  "Czech Republic"],
    "B": ["Canada",        "Bosnia and Herzegovina", "Qatar",        "Switzerland"],
    "C": ["Brazil",        "Morocco",               "Haiti",        "Scotland"],
    "D": ["United States", "Paraguay",              "Australia",    "Turkey"],
    "E": ["Germany",       "Curaçao",               "Ivory Coast",  "Ecuador"],
    "F": ["Netherlands",   "Japan",                 "Sweden",       "Tunisia"],
    "G": ["Belgium",       "Egypt",                 "Iran",         "New Zealand"],
    "H": ["Spain",         "Cape Verde",            "Saudi Arabia", "Uruguay"],
    "I": ["France",        "Senegal",               "Iraq",         "Norway"],
    "J": ["Argentina",     "Algeria",               "Austria",      "Jordan"],
    "K": ["Portugal",      "DR Congo",              "Uzbekistan",   "Colombia"],
    "L": ["England",       "Croatia",               "Ghana",        "Panama"],    
}

# Results already played (from live data)
PLAYED = {(norm(r["home_team"]), norm(r["away_team"])): (int(r["home_score"]), int(r["away_score"]))
          for _, r in live.iterrows()}

print(f"  Matches already played: {len(PLAYED)}")
print("  Caching all pairwise probabilities …")

CACHE = {}
for ht in WC_TEAMS:
    for at in WC_TEAMS:
        if ht == at: continue
        ph_p, pd_p, pa_p, lh, la = pprobs(ht, at, neutral=True)
        ph_c, pd_c, pa_c         = cprobs(ht, at, neutral=True)
        eh = ELO.get(ht,1500.0); ea = ELO.get(at,1500.0)
        ew = exp_s(eh, ea)
        ep = (ew*0.85, 0.15, (1-ew)*0.85)
        ph  = 0.40*ph_p + 0.40*ph_c + 0.20*ep[0]
        pd_ = 0.40*pd_p + 0.40*pd_c + 0.20*ep[1]
        pa  = 0.40*pa_p + 0.40*pa_c + 0.20*ep[2]
        tot = ph+pd_+pa
        CACHE[(ht,at)] = {
            "p_home": ph/tot, "p_draw": pd_/tot, "p_away": pa/tot,
            "lam_h": lh, "lam_a": la
        }
print("  Done ✓")


# =============================================================
# PHASE 7 — MONTE CARLO SIMULATION
# =============================================================

print("\n" + "=" * 60)
print(f"PHASE 7 — Monte Carlo ({N_SIMULATIONS:,} simulations)")
print("=" * 60)

# ---- Track REAL knockout eliminations --------------------------------
# GS_SET = every group-stage fixture pair, both directions.
# Any match in PLAYED that is NOT in GS_SET must be a real knockout
# result already played — so the loser is truly out of the tournament,
# no matter how our simplified bracket pairs them in simulation.
GS_SET = set(GS_FIXTURES) | {(a, h) for h, a in GS_FIXTURES}

ELIMINATED = set()
for (ht, at), (hg, ag) in PLAYED.items():
    if (ht, at) not in GS_SET:          # knockout match, not group stage
        if hg > ag:
            ELIMINATED.add(at)
        elif ag > hg:
            ELIMINATED.add(ht)
        # (a real KO draw shouldn't happen post-penalties, but if the
        #  sheet stores 90-min score only, you may need to add a
        #  'winner' column instead of inferring from score)

print(f"  Real knockout losers already eliminated: {len(ELIMINATED)}")
if ELIMINATED:
    print(f"    {sorted(ELIMINATED)}")


def sim_match(ht, at):
    """Returns (winner/'draw', hg, ag). Uses real result if already played."""
    # Check if this exact fixture exists in your live data
    if (ht, at) in PLAYED:
        hg, ag = PLAYED[(ht, at)]
        return (ht if hg > ag else (at if ag > hg else "draw")), hg, ag
    elif (at, ht) in PLAYED:
        ag, hg = PLAYED[(at, ht)]
        return (at if ag > hg else (ht if hg > ag else "draw")), hg, ag
    
    # Otherwise, simulate via cached Poisson lambdas
    p  = CACHE[(ht, at)]
    hg = np.random.poisson(p["lam_h"])
    ag = np.random.poisson(p["lam_a"])
    if hg > ag: return ht, hg, ag
    if hg < ag: return at, hg, ag
    return "draw", hg, ag


def sim_ko(ht, at):
    """
    Simulates a knockout match. 
    Guarantees that already eliminated teams automatically lose.
    """
    # Hard rule: If a team is already realistically eliminated, they CANNOT advance
    if ht in ELIMINATED and at in ELIMINATED:
        # If both are somehow eliminated (rare simulation quirk), higher ELO survives
        return ht if ELO.get(ht, 1500) >= ELO.get(at, 1500) else at
    if ht in ELIMINATED:
        return at  # 'at' advances automatically because 'ht' is already out
    if at in ELIMINATED:
        return ht  # 'ht' advances automatically because 'at' is already out

    # If both are still alive, play the match normally
    w, hg, ag = sim_match(ht, at)
    if w == "draw":
        # Check if the live sheet has a record of who won the penalty shootout
        # (Assuming you might have a way to check, otherwise fallback to probability weights)
        p   = CACHE[(ht, at)]
        tot = p["p_home"] + p["p_away"]
        w   = np.random.choice([ht, at], p=[p["p_home"]/tot, p["p_away"]/tot])
    return w




def sim_group(grp_name, teams):
    pts = {t: 0 for t in teams}; gd = {t: 0 for t in teams}; gf = {t: 0 for t in teams}
    for ht, at in itertools.combinations(teams, 2):
        w, hg, ag = sim_match(ht, at)
        gd[ht] += hg - ag; gd[at] += ag - hg
        gf[ht] += hg;      gf[at] += ag
        if w == ht:   pts[ht] += 3
        elif w == at: pts[at] += 3
        else:         pts[ht] += 1; pts[at] += 1
    ranked = sorted(teams, key=lambda t: (pts[t], gd[t], gf[t], ELO.get(t, 1500)), reverse=True)
    return ranked, {t: {"pts": pts[t], "gd": gd[t], "gf": gf[t]} for t in teams}


def sim_tournament():
    group_keys = list(GROUPS.keys())
    ranked_all = {}
    stats_all  = {}
    
    # 1. Run Group Stage Simulations
    for g in group_keys:
        ranked, stats = sim_group(g, GROUPS[g])
        ranked_all[g] = ranked
        stats_all[g]  = stats

    # 2. Separate qualifiers systematically
    winners = {g: ranked_all[g][0] for g in group_keys}
    runners_up = {g: ranked_all[g][1] for g in group_keys}
    
    thirds = []
    for g in group_keys:
        third = ranked_all[g][2]
        thirds.append((third, stats_all[g][third]["pts"], stats_all[g][third]["gd"], stats_all[g][third]["gf"], ELO.get(third, 1500)))
    best8 = [t[0] for t in sorted(thirds, key=lambda x: (x[1], x[2], x[3], x[4]), reverse=True)[:8]]

    # 3. Construct a standard Round of 32 pairing tree
    # This prevents group partners (like A1 and A2) from facing each other immediately
    r32_pairings = [
        (winners["A"], runners_up["B"]), (winners["C"], runners_up["D"]),
        (winners["E"], runners_up["F"]), (winners["G"], runners_up["H"]),
        (winners["I"], runners_up["J"]), (winners["K"], runners_up["L"]),
        (winners["B"], winners["D"]),     (winners["F"], winners["H"]),
        
        # Mix the remaining runners up with the best 3rd places
        (winners["J"], best8[0]),         (winners["L"], best8[1]),
        (runners_up["A"], best8[2]),      (runners_up["C"], best8[3]),
        (runners_up["E"], best8[4]),      (runners_up["G"], best8[5]),
        (runners_up["I"], best8[6]),      (runners_up["K"], best8[7])
    ]

    # 4. Progress through Knockout Rounds
    remaining = []
    for ht, at in r32_pairings:
        remaining.append(sim_ko(ht, at)) # Automatically drops real-world losers

    sf_teams  = []
    fin_teams = []
    round_num = 1 # We just finished R32 (Round 1 of KO)
    
    while len(remaining) > 1:
        round_num += 1
        nxt = []
        for i in range(0, len(remaining), 2):
            w = sim_ko(remaining[i], remaining[i+1])
            nxt.append(w)
            
            if round_num == 3:   # Quarter-Final winners
                sf_teams.append(w)
            if round_num == 4:   # Semi-Final winners
                fin_teams.append(w)
                
        remaining = nxt

    winner = remaining[0]
    return winner, sf_teams, fin_teams

print("  Simulating … ", end="", flush=True)
counts = {t: {"sf": 0, "final": 0, "win": 0} for t in WC_TEAMS}

for i in range(N_SIMULATIONS):
    if i % 2000 == 0: print(f"{i//1000}k ", end="", flush=True)
    winner, sf, fin = sim_tournament()
    if winner in counts: counts[winner]["win"] += 1
    for t in fin:
        if t in counts: counts[t]["final"] += 1
    for t in sf:
        if t in counts: counts[t]["sf"] += 1

print("done ✓")


# PHASE 8 — RESULTS + OUTPUT CSVs

print("\n" + "=" * 60)
print("PHASE 8 — Results & CSV outputs")
print("=" * 60)

results = []
for t in WC_TEAMS:
    c = counts[t]
    results.append({
        "team":          t,
        "elo":           round(ELO.get(t,1500),1),
        "p_semifinal":   round(c["sf"]/N_SIMULATIONS*100,2),
        "p_final":       round(c["final"]/N_SIMULATIONS*100,2),
        "p_winner":      round(c["win"]/N_SIMULATIONS*100,2),
    })

res_df = pd.DataFrame(results).sort_values("p_winner", ascending=False).reset_index(drop=True)
res_df.index += 1; res_df.index.name = "rank"

print(f"\n  {'Team':<28} {'Elo':>6}  {'Semi%':>7}  {'Final%':>7}  {'Win%':>7}")
print("  " + "-"*60)
for _, r in res_df.iterrows():
    print(f"  {r['team']:<28} {r['elo']:>6.0f}  {r['p_semifinal']:>6.1f}%  {r['p_final']:>6.1f}%  {r['p_winner']:>6.1f}%")

# Save CSVs
res_df.to_csv("output/wc2026_predictions.csv")
print("\n  ✓ output/wc2026_predictions.csv")

# Group stage match predictions
match_preds = []
for ht,at in GS_FIXTURES:
    if (ht,at) in PLAYED:
        hg,ag = PLAYED[(ht,at)]
        match_preds.append({
            "home_team":ht,"away_team":at,
            "status":"PLAYED",
            "actual_score":f"{hg}-{ag}",
            "p_home_win":"—","p_draw":"—","p_away_win":"—",
            "predicted_winner":ht if hg>ag else (at if ag>hg else "Draw"),
        })
    else:
        p = CACHE.get((ht,at), CACHE.get((at,ht),{}))
        if not p: continue
        pw = ht if p["p_home"]>p["p_away"] else (at if p["p_away"]>p["p_home"] else "Draw")
        match_preds.append({
            "home_team":ht,"away_team":at,
            "status":"UPCOMING",
            "actual_score":"TBD",
            "p_home_win":  round(p["p_home"]*100,1),
            "p_draw":      round(p["p_draw"]*100,1),
            "p_away_win":  round(p["p_away"]*100,1),
            "predicted_winner": pw,
        })

pd.DataFrame(match_preds).to_csv("output/group_stage_predictions.csv", index=False)
print("  ✓ output/group_stage_predictions.csv")

# Elo ratings
elo_df = pd.DataFrame([{"team":t,"elo":round(v,1)} for t,v in
                        sorted(ELO.items(),key=lambda x:-x[1])])
elo_df.to_csv("output/elo_ratings.csv", index=False)
print("  ✓ output/elo_ratings.csv")

print("\n" + "=" * 60)
print("COMPLETE")
print("=" * 60)
print(f"\n  🏆 Predicted winner  : {res_df.iloc[0]['team']}  ({res_df.iloc[0]['p_winner']:.1f}%)")
print(f"  🥈 Runner-up        : {res_df.iloc[1]['team']}  ({res_df.iloc[1]['p_winner']:.1f}%)")
print(f"  🥉 Third most likely: {res_df.iloc[2]['team']}  ({res_df.iloc[2]['p_winner']:.1f}%)")
print("\n  All CSVs saved to ./output/ — import directly into Power BI.")