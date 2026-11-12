# -*- coding: utf-8 -*-
"""
반도체 공정 데이터 모델링 2종 (작성: 조은숙)
  A. AlCu PVD 웨이퍼 불량 분류  — 시간 순서 분할, 데이터 누수 차단
  B. LAM 9600 식각 이상 탐지    — 정상 웨이퍼만 학습하는 PCA T²/SPE
사용법: python semi_models.py <데이터 최상위 폴더> <그림 저장 폴더>
"""
import sys, os, glob, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, KFold

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
OUT = sys.argv[2] if len(sys.argv) > 2 else "figs"
os.makedirs(OUT, exist_ok=True)
for name in ["Noto Sans CJK KR", "Malgun Gothic", "AppleGothic", "NanumGothic"]:
    if any(name in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = name; break
plt.rcParams.update({"axes.unicode_minus": False, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "axes.axisbelow": True, "grid.color": "#e6e5e0", "grid.linewidth": .6,
                     "axes.edgecolor": "#8a8984", "axes.titleweight": "bold", "axes.titlesize": 12, "font.size": 10})
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GRAY = "#9b9a95"
SEED = 42
R = {}
folder = lambda p: glob.glob(os.path.join(ROOT, p + "_*"))[0]

# =====================================================================
# A. AlCu PVD 불량 분류
# =====================================================================
d = pd.read_csv(glob.glob(os.path.join(folder("1"), "*.csv"))[0], sep=";", skiprows=1, low_memory=False)
d["runstart"] = pd.to_datetime(d["runstart"])
d = d.sort_values("runstart").reset_index(drop=True)
y = d["quality_overall"].astype(int)

# 누수 차단: 측정 결과(RAW_VALUE_*, 두께·저항 평균, 품질 판정)와 식별자는 입력에서 제외
drop = [c for c in d.columns if c.startswith("RAW_VALUE") or c.startswith("quality")] + \
       ["mean_thickness", "mean_resistance", "bat_id", "wafer_id", "lot", "batchstart", "batchend", "runstart", "runend"]
X = d.drop(columns=drop)
X = pd.get_dummies(X, columns=["equipment", "machine"], dtype=float)
X = X.loc[:, X.nunique() > 1]                       # 값이 하나뿐인 열 제거

# 시간 순서 분할: 앞 75% 학습, 뒤 25% 시험 (미래 데이터로 과거를 학습하지 않음)
cut = int(len(d) * .75)
Xtr, Xte, ytr, yte = X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]
models = {
    "로지스틱 회귀": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                               LogisticRegression(max_iter=3000, class_weight="balanced", C=.1)),
    "랜덤 포레스트": make_pipeline(SimpleImputer(strategy="median"),
                               RandomForestClassifier(500, min_samples_leaf=3, class_weight="balanced_subsample",
                                                      n_jobs=-1, random_state=SEED)),
    "그래디언트 부스팅": HistGradientBoostingClassifier(max_iter=400, learning_rate=.05, max_leaf_nodes=15,
                                                  class_weight="balanced", random_state=SEED),
}
res, probs = {}, {}
for nm, m in models.items():
    m.fit(Xtr, ytr); p = m.predict_proba(Xte)[:, 1]; probs[nm] = p
    # 상위 10% 위험 웨이퍼만 검사한다고 할 때 잡아내는 불량 비율
    k = int(len(p) * .10); top = np.argsort(-p)[:k]
    res[nm] = dict(roc_auc=round(roc_auc_score(yte, p), 3), pr_auc=round(average_precision_score(yte, p), 3),
                   recall_top10=round(yte.iloc[top].sum() / yte.sum(), 3),
                   precision_top10=round(yte.iloc[top].mean(), 3))
best = "랜덤 포레스트"                                 # ROC-AUC 최고, PR-AUC 동률
pi = permutation_importance(models[best], Xte, yte, scoring="average_precision", n_repeats=10,
                            random_state=SEED, n_jobs=-1)
imp = pd.Series(pi.importances_mean, index=X.columns).sort_values(ascending=False).head(10)

# 비교: 시간 무시 무작위 5겹 교차검증 (낙관적으로 나오는지 확인)
rnd = []
for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y):
    m = models[best]; m.fit(X.iloc[tr], y.iloc[tr]); p = m.predict_proba(X.iloc[te])[:, 1]
    rnd.append([roc_auc_score(y.iloc[te], p), average_precision_score(y.iloc[te], p), y.iloc[te].mean()])
rnd = np.array(rnd).mean(0)
fig, ax = plt.subplots(1, 3, figsize=(17, 4.6), gridspec_kw={"width_ratios": [1, 1, 1.3]})
base = yte.mean()
for i, (nm, p) in enumerate(probs.items()):
    pr, rc, _ = precision_recall_curve(yte, p)
    ax[0].plot(rc, pr, color=C[i], lw=2, label=f"{nm} (PR-AUC {res[nm]['pr_auc']:.2f})")
ax[0].axhline(base, color=GRAY, ls="--", lw=1, label=f"무작위 기준 ({base:.3f})")
ax[0].set_xlabel("재현율 (불량 중 잡아낸 비율)"); ax[0].set_ylabel("정밀도 (경보 중 실제 불량 비율)")
ax[0].set_title("정밀도-재현율 곡선 (시험 기간)"); ax[0].legend(fontsize=8, frameon=False); ax[0].set_ylim(0, 1.02)
lab = ["ROC-AUC", "PR-AUC ÷ 불량률\n(무작위 대비 배수)"]
v_rnd = [rnd[0], rnd[1] / rnd[2]]; v_time = [res[best]["roc_auc"], res[best]["pr_auc"] / base]
xs = np.arange(2); w = .36
ax[1].bar(xs - w / 2, v_rnd, w, color="#a9c8ef", label="무작위 5겹 교차검증")
ax[1].bar(xs + w / 2, v_time, w, color=C[0], label="시간 순서 분할 (실제 상황)")
for i in range(2):
    ax[1].text(xs[i] - w / 2, v_rnd[i] + .15, f"{v_rnd[i]:.2f}", ha="center", fontsize=9)
    ax[1].text(xs[i] + w / 2, v_time[i] + .15, f"{v_time[i]:.2f}", ha="center", fontsize=9)
ax[1].set_xticks(xs); ax[1].set_xticklabels(lab); ax[1].set_ylim(0, max(v_rnd) * 1.25)
ax[1].set_title(f"평가 방식에 따른 성능 차이 ({best})"); ax[1].legend(fontsize=8, frameon=False, loc="upper left")
ax[2].barh(imp.index[::-1], imp.values[::-1], color=C[0], height=.6)
ax[2].set_title(f"순열 중요도 Top 10 ({best})"); ax[2].set_xlabel("PR-AUC 감소량"); ax[2].tick_params(axis="y", labelsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figA_alcu_model.png"), dpi=130, bbox_inches="tight", facecolor="white"); plt.close(fig)
R["A"] = dict(random_cv=dict(roc_auc=round(rnd[0], 3), pr_auc=round(rnd[1], 3), base=round(rnd[2], 3)),n_features=X.shape[1], train=[len(Xtr), int(ytr.sum()), round(ytr.mean() * 100, 2)],
              test=[len(Xte), int(yte.sum()), round(yte.mean() * 100, 2)],
              split_date=str(d.runstart.iloc[cut].date()), results=res, best=best,
              importance={k: round(v, 4) for k, v in imp.items()})

# =====================================================================
# B. LAM 9600 이상 탐지
# =====================================================================
p2 = folder("2")
ts = pd.read_csv(os.path.join(p2, "lam9600_machine_timeseries.csv"))
meta = pd.read_csv(os.path.join(p2, "lam9600_wafer_metadata.csv"))
short = meta.loc[meta.n_timesteps < 50, "wafer_id"].tolist()
ts = ts[~ts.wafer_id.isin(short)]
sens = [c for c in ts.columns if c not in ("wafer_id", "split", "is_fault", "fault_type", "time", "step_number")]
# 특징: 공정 단계(step 4, 5)별 평균·표준편차 → 시간 구조를 일부 보존
F = ts.groupby(["wafer_id", "step_number"])[sens].agg(["mean", "std"]).unstack("step_number")
F.columns = [f"{s}_{a}_s{int(st)}" for s, a, st in F.columns]
F = F.fillna(F.median())
F = F.loc[:, F.std() > 0]
info = meta.set_index("wafer_id").loc[F.index]
info["exp"] = info.index.str[:3]                      # l29 / l31 / l33 = 실험 차수
normal_ids = info.index[info.is_fault == 0]; fault_ids = info.index[info.is_fault == 1]

N_COMP, Q = .80, .95          # PCA 설명분산 80%, 관리한계 = 정상 점수의 95% 분위

def fit_model(train):
    sc = StandardScaler().fit(train); Z = sc.transform(train)
    pca = PCA(N_COMP, random_state=SEED).fit(Z)
    return sc, pca

def scores(model, X):
    """웨이퍼별 이상 점수 3종: 단변량 max|z|, PCA T², PCA SPE(잔차 제곱합)"""
    sc, pca = model; Z = sc.transform(X); T = pca.transform(Z)
    return np.c_[np.abs(Z).max(1), (T ** 2 / pca.explained_variance_).sum(1),
                 ((Z - pca.inverse_transform(T)) ** 2).sum(1)]

def oof_limits(X):
    """관리한계를 학습 데이터가 아닌 교차검증 바깥(out-of-fold) 정상 점수로 정함 → 과적합 방지"""
    oof = np.zeros((len(X), 3))
    for tr, te in KFold(5, shuffle=True, random_state=SEED).split(X):
        oof[te] = scores(fit_model(X.iloc[tr]), X.iloc[te])
    return np.quantile(oof, Q, axis=0), np.quantile(oof, 1 - (1 - Q) / 2, axis=0)

def flags(sc_, lim, lim2):
    return {"단변량 max|z|": sc_[:, 0] > lim[0], "PCA T²": sc_[:, 1] > lim[1], "PCA SPE": sc_[:, 2] > lim[2],
            "T² 또는 SPE": (sc_[:, 1] > lim2[1]) | (sc_[:, 2] > lim2[2])}

def evaluate(per_exp):
    """정상: 중첩 5겹 교차검증으로 오경보율 / 불량: 정상 전체로 학습한 모델로 탐지율"""
    groups = info.groupby("exp").groups if per_exp else {"all": info.index}
    keys = ["단변량 max|z|", "PCA T²", "PCA SPE", "T² 또는 SPE"]
    nf = {k: pd.Series(False, index=normal_ids) for k in keys}
    ff = {k: pd.Series(False, index=fault_ids) for k in keys}
    fs = pd.DataFrame(index=fault_ids, columns=["t2", "spe"], dtype=float)
    ns = pd.DataFrame(index=normal_ids, columns=["t2", "spe"], dtype=float)
    for g, idx in groups.items():
        nid = [i for i in idx if i in normal_ids]; fid = [i for i in idx if i in fault_ids]
        Xn = F.loc[nid]
        for tr, te in KFold(5, shuffle=True, random_state=SEED).split(Xn):
            inner = Xn.iloc[tr]; lim, lim2 = oof_limits(inner)
            s_ = scores(fit_model(inner), Xn.iloc[te]); ids = Xn.index[te]
            ns.loc[ids, "t2"], ns.loc[ids, "spe"] = s_[:, 1] / lim2[1], s_[:, 2] / lim2[2]
            for k, v in flags(s_, lim, lim2).items(): nf[k].loc[ids] = v
        if fid:
            lim, lim2 = oof_limits(Xn); s_ = scores(fit_model(Xn), F.loc[fid])
            fs.loc[fid, "t2"], fs.loc[fid, "spe"] = s_[:, 1] / lim2[1], s_[:, 2] / lim2[2]
            for k, v in flags(s_, lim, lim2).items(): ff[k].loc[fid] = v
    table = {k: dict(false_alarm=round(nf[k].mean() * 100, 1), detection=round(ff[k].mean() * 100, 1),
                     detected=int(ff[k].sum())) for k in keys}
    return table, ff["T² 또는 SPE"], fs, ns

tab_all, _, _, _ = evaluate(per_exp=False)
tab_exp, flag_exp, fs, ns = evaluate(per_exp=True)

fig, ax = plt.subplots(1, 3, figsize=(17, 4.8), gridspec_kw={"width_ratios": [1.1, 1.2, 1]})
Zall = PCA(2).fit_transform(StandardScaler().fit_transform(F.loc[normal_ids]))
for i, e in enumerate(sorted(info.exp.unique())):
    mk = info.loc[normal_ids, "exp"].values == e
    ax[0].scatter(Zall[mk, 0], Zall[mk, 1], s=18, color=C[i], label=f"실험 {e}")
ax[0].set_title("정상 웨이퍼 PCA — 실험 차수별 색"); ax[0].set_xlabel("PC1"); ax[0].set_ylabel("PC2")
ax[0].legend(fontsize=8, frameon=False)
ax[1].scatter(ns.t2, ns.spe, s=16, color=GRAY, label="정상 (교차검증)")
ax[1].scatter(fs.t2, fs.spe, s=34, color=C[7], edgecolor="white", lw=.8, label="불량")
for w in fs.index:
    ax[1].annotate(info.loc[w, "fault_type"], (fs.loc[w, "t2"], fs.loc[w, "spe"]), fontsize=6.5,
                   xytext=(3, 2), textcoords="offset points", color="#444")
ax[1].axvline(1, color="#555", ls="--", lw=.8); ax[1].axhline(1, color="#555", ls="--", lw=.8)
ax[1].set_xscale("log"); ax[1].set_yscale("log")
ax[1].set_xlabel("T² ÷ 관리한계 (log)"); ax[1].set_ylabel("SPE ÷ 관리한계 (log)")
ax[1].set_title("실험별 PCA 모델의 T²·SPE (점선 밖 = 경보)"); ax[1].legend(fontsize=8, frameon=False, loc="lower right")
meths = list(tab_exp); xs = np.arange(len(meths)); w = .36
a1 = [tab_all[m]["detection"] for m in meths]; a2 = [tab_exp[m]["detection"] for m in meths]
ax[2].bar(xs - w / 2, a1, w, color="#a9c8ef", label="전체 정상 1개 모델")
ax[2].bar(xs + w / 2, a2, w, color=C[0], label="실험별 모델")
for i in range(len(meths)):
    ax[2].text(xs[i] - w / 2, a1[i] + 1.5, f"{a1[i]:.0f}", ha="center", fontsize=8)
    ax[2].text(xs[i] + w / 2, a2[i] + 1.5, f"{a2[i]:.0f}", ha="center", fontsize=8)
ax[2].set_xticks(xs); ax[2].set_xticklabels(meths, fontsize=8.5); ax[2].set_ylim(0, 105)
ax[2].set_ylim(0, 125); ax[2].set_yticks(range(0, 101, 20)); ax[2].set_ylabel("불량 탐지율 (%)"); ax[2].set_title("방법별 불량 탐지율 (불량 21장)\n관리한계: 정상의 교차검증 점수 기준"); ax[2].legend(fontsize=8, frameon=False, loc="upper center", ncol=2)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figB_lam_anomaly.png"), dpi=130, bbox_inches="tight", facecolor="white"); plt.close(fig)
R["B"] = dict(n_features=F.shape[1], normals=len(normal_ids), faults=len(fault_ids), excluded=short,
              exp_counts=info.groupby(["exp", "is_fault"]).size().unstack(fill_value=0).to_dict(),
              all=tab_all, per_exp=tab_exp,
              missed=info.loc[flag_exp.index[~flag_exp.values], "fault_type"].tolist(),
              caught=info.loc[flag_exp.index[flag_exp.values], "fault_type"].tolist())
json.dump(R, open(os.path.join(OUT, "model_results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(R, ensure_ascii=False, indent=1))
