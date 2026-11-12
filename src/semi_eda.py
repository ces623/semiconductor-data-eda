# -*- coding: utf-8 -*-
"""
반도체 공정 공개 데이터 9종 탐색적 분석(EDA) · 시각화 코드 (작성: 조은숙)
사용법:  python semi_eda.py  <데이터 최상위 폴더>  <그림 저장 폴더>
  예)    python semi_eda.py  ./반도체데이터  ./figs
필요 라이브러리: pandas, numpy, matplotlib, scikit-learn, Pillow, openpyxl
"""
import sys, os, glob, json, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from PIL import Image

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
OUT = sys.argv[2] if len(sys.argv) > 2 else "figs"
os.makedirs(OUT, exist_ok=True)

# ---------- 공통 스타일 (한글 폰트 + 색상 팔레트) ----------
for name in ["Noto Sans CJK KR", "Malgun Gothic", "AppleGothic", "NanumGothic"]:
    if any(name in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = name
        break
plt.rcParams.update({"axes.unicode_minus": False, "figure.dpi": 110,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "axes.axisbelow": True, "grid.color": "#e6e5e0", "grid.linewidth": .6,
                     "axes.edgecolor": "#8a8984", "axes.titleweight": "bold",
                     "axes.titlesize": 12, "font.size": 10})
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GRAY = "#9b9a95"
STATS = {}

def folder(prefix):
    return glob.glob(os.path.join(ROOT, prefix + "_*"))[0]

def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)

# ======================================================================
# 1. 웨이퍼 단위 요약표 : Infineon AlCu PVD (APC/SPC)
# ======================================================================
f = glob.glob(os.path.join(folder("1"), "*.csv"))[0]
d1 = pd.read_csv(f, sep=";", skiprows=1, low_memory=False)      # 1행은 제목줄 → 건너뜀
d1["runstart"] = pd.to_datetime(d1["runstart"])
d1["month"] = d1["runstart"].dt.to_period("M").dt.to_timestamp()
fig, ax = plt.subplots(1, 3, figsize=(16, 4.4), gridspec_kw={"width_ratios": [2.2, 1, 1.3]})
m = d1.groupby("month")["mean_thickness"].agg(["mean", "std"])
ax[0].scatter(d1["runstart"], d1["mean_thickness"], s=3, color=GRAY, alpha=.35, label="웨이퍼 1장")
ax[0].plot(m.index, m["mean"], color=C[0], lw=2, label="월 평균")
bad = d1[d1["quality_overall"] == 1]
ax[0].scatter(bad["runstart"], bad["mean_thickness"], s=8, color=C[7], label="불량(quality_overall=1)")
ax[0].set_title("① AlCu 막 두께 추이 (2021-07 ~ 2025-05)"); ax[0].set_ylabel("mean_thickness")
ax[0].legend(loc="upper left", fontsize=8, frameon=False)
r = d1.groupby("machine")["quality_overall"].mean().sort_index() * 100
ax[1].bar(r.index.str.replace("machine_", "M"), r.values, color=C[0], width=.6)
for i, v in enumerate(r.values):
    ax[1].text(i, v + .2, f"{v:.1f}%", ha="center", fontsize=8)
ax[1].set_title("② 장비(machine)별 불량률"); ax[1].set_ylabel("불량률 (%)")
num = d1.select_dtypes("number").drop(columns=["mean_thickness", "mean_resistance", "quality_overall",
                                                "stable_state", "bat_id", "slot"], errors="ignore")
num = num.loc[:, ~num.columns.str.startswith("RAW_VALUE")]          # 측정 원값(결과)은 제외 → 공정 입력만
corr = num.corrwith(d1["mean_thickness"]).dropna()
top = corr.reindex(corr.abs().sort_values(ascending=False).index[:8])[::-1]
ax[2].barh(top.index, top.values, color=[C[0] if v > 0 else C[1] for v in top.values], height=.6)
ax[2].axvline(0, color="#555", lw=.8); ax[2].set_title("③ 두께와 상관 높은 공정 변수 Top 8")
ax[2].set_xlabel("피어슨 상관계수"); ax[2].tick_params(axis="y", labelsize=7.5)
save(fig, "fig1_alcu.png")
STATS["1"] = dict(rows=len(d1), cols=d1.shape[1], machines=d1.machine.nunique(),
                  equipment=d1.equipment.nunique(), bad_rate=round(d1.quality_overall.mean() * 100, 2),
                  thick_mean=round(d1.mean_thickness.mean(), 2), thick_std=round(d1.mean_thickness.std(), 2),
                  top_corr={k: round(v, 3) for k, v in top[::-1].items()},
                  bad_by_machine={k: round(v, 2) for k, v in r.items()},
                  period=[str(d1.runstart.min().date()), str(d1.runstart.max().date())])

# ======================================================================
# 2. 센서 시계열 : LAM 9600 금속 식각 (정상 108 / 불량 21)
# ======================================================================
p2 = folder("2")
ts = pd.read_csv(os.path.join(p2, "lam9600_machine_timeseries.csv"))
meta = pd.read_csv(os.path.join(p2, "lam9600_wafer_metadata.csv"))
short = meta.loc[meta.n_timesteps < 50, "wafer_id"].tolist()      # 기록이 비정상적으로 짧은 웨이퍼 제외
ts = ts[~ts.wafer_id.isin(short)].copy()
ts["t"] = ts.groupby("wafer_id").cumcount()          # 웨이퍼별 경과 샘플 번호
sensors = [("pressure", "챔버 압력"), ("bcl3_flow", "BCl3 유량"),
           ("rf_tuner", "RF 튜너 위치"), ("tcp_top_pwr", "TCP 상부 전력")]
fig, ax = plt.subplots(1, 5, figsize=(18, 3.8), gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.25]})
for k, (s, kr) in enumerate(sensors):
    q = ts[ts.is_fault == 0].groupby("t")[s].quantile([.05, .5, .95]).unstack()
    q = q[q.index <= 100]
    ax[k].fill_between(q.index, q[.05], q[.95], color="#d9e6f5", label="정상 5~95% 범위")
    ax[k].plot(q.index, q[.5], color=C[0], lw=1.6, label="정상 중앙값")
    for wid, g in ts[ts.is_fault == 1].groupby("wafer_id"):
        g = g[g.t <= 100]
        ax[k].plot(g["t"], g[s], color=C[7], lw=.6, alpha=.55)
    ax[k].set_title(f"{kr} ({s})", fontsize=10); ax[k].set_xlabel("시간 인덱스")
ax[0].plot([], [], color=C[7], label="불량 웨이퍼")
ax[0].legend(fontsize=7.5, frameon=False)
feat_cols = [c for c in ts.columns if c not in ("wafer_id", "split", "is_fault", "fault_type", "time", "step_number", "t")]
agg = ts.groupby("wafer_id")[feat_cols].agg(["mean", "std"]).fillna(0)
agg.columns = ["_".join(c) for c in agg.columns]
lab = ts.groupby("wafer_id")["is_fault"].first().reindex(agg.index)
Z = PCA(2).fit_transform(StandardScaler().fit_transform(agg))
ax[4].scatter(Z[lab == 0, 0], Z[lab == 0, 1], s=14, color=GRAY, label="정상")
ax[4].scatter(Z[lab == 1, 0], Z[lab == 1, 1], s=26, color=C[7], edgecolor="white", lw=.8, label="불량")
ax[4].set_title("웨이퍼별 요약특징 PCA", fontsize=10); ax[4].set_xlabel("PC1"); ax[4].set_ylabel("PC2")
ax[4].legend(fontsize=8, frameon=False)
save(fig, "fig2_lam9600.png")
# 정상 대비 불량 웨이퍼의 평균값 차이(z) 큰 센서
zdiff = ((agg[lab == 1].mean() - agg[lab == 0].mean()) / agg[lab == 0].std()).abs().sort_values(ascending=False)
STATS["2"] = dict(excluded=short, wafers=len(meta), normal=int((meta.is_fault == 0).sum()), fault=int(meta.is_fault.sum()),
                  rows=len(ts), sensors=len(feat_cols), fault_types=meta.fault_type.nunique() - 1,
                  top_diff={k: round(v, 2) for k, v in zdiff.head(5).items()})

# ======================================================================
# 3. 위치별 계측 : Fraunhofer Bosch 식각 (웨이퍼당 89 지점)
# ======================================================================
d3 = pd.read_csv(glob.glob(os.path.join(folder("3"), "*.csv"))[0])
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), gridspec_kw={"width_ratios": [1, 1, 1.5]})
pm = d3.groupby(["X", "Y"])[["si_etch", "oxide_etch"]].mean().reset_index()
for k, (col, ttl, cmap) in enumerate([("si_etch", "Si 식각 깊이 평균 맵", "Blues"),
                                       ("oxide_etch", "산화막(마스크) 식각량 평균 맵", "Oranges")]):
    sc = ax[k].scatter(pm.X / 1000, pm.Y / 1000, c=pm[col], s=95, marker="s", cmap=cmap, edgecolor="white")
    ax[k].add_patch(plt.Circle((0, 0), 100, fill=False, color="#888", lw=1))
    ax[k].set_aspect("equal"); ax[k].set_xlim(-105, 105); ax[k].set_ylim(-105, 105)
    ax[k].set_title(ttl); ax[k].set_xlabel("X (mm)"); ax[k].set_ylabel("Y (mm)"); ax[k].grid(False)
    plt.colorbar(sc, ax=ax[k], shrink=.8, label=col)
wm = d3.groupby(["lot_number", "wafer_number"])["si_etch"].mean().reset_index()   # experiment_key = 날짜_웨이퍼
for lot, g in wm.groupby("lot_number"):
    ax[2].plot(g.wafer_number, g.si_etch, color=GRAY, lw=1, marker="o", ms=2.5, alpha=.7)
mw = wm.groupby("wafer_number").si_etch.mean()
ax[2].plot(mw.index, mw.values, color=C[0], lw=2.5, marker="o", ms=5, label="10개 로트 평균")
ax[2].plot([], [], color=GRAY, label="로트별 (Lot 1~10)")
ax[2].set_title("로트 내 웨이퍼 처리 순서별 Si 식각 깊이 (드리프트)"); ax[2].set_xlabel("웨이퍼 처리 순서")
ax[2].set_ylabel("웨이퍼 평균 si_etch"); ax[2].set_xticks(range(1, 11)); ax[2].legend(fontsize=8, frameon=False)
save(fig, "fig3_bosch.png")
wstd = d3.groupby(["lot_number", "wafer_number"])["si_etch"].agg(["mean", "std"])
d3["r"] = np.hypot(d3.X, d3.Y) / 1000
STATS["3"] = dict(rows=len(d3), wafers=len(wm), lots=wm.lot_number.nunique(), first_last=[round(mw.iloc[0],2), round(mw.iloc[-1],2)],
                  si_mean=round(d3.si_etch.mean(), 2), si_min=round(d3.si_etch.min(), 2), si_max=round(d3.si_etch.max(), 2),
                  within_wafer_cv=round((wstd["std"] / wstd["mean"]).mean() * 100, 2),
                  edge_vs_center=round(d3[d3.r > 70].si_etch.mean() - d3[d3.r < 30].si_etch.mean(), 2),
                  selectivity=round((d3.si_etch / d3.oxide_etch).mean(), 1),
                  nan_postox=int(d3.postox_thickness_nan.isna().sum()))

# ======================================================================
# 4. 스펙트럼·프로파일 : ToF-SIMS WOx (18O 표지층)
# ======================================================================
p4 = folder("4"); ex = os.path.join(p4, "ex")
if not os.path.isdir(ex):
    import zipfile; zipfile.ZipFile(os.path.join(p4, "exports_WOx.zip")).extractall(ex)
prof = sorted(glob.glob(os.path.join(ex, "**", "depth_profiles", "*_r1*.tsv"), recursive=True))
fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
nominal = {"S3": 50, "S1": 25, "S2": 16.7, "S4": 12.5}
order = sorted(prof, key=lambda x: -nominal[os.path.basename(x)[:2]])
s4 = {}
for i, f in enumerate(order):
    t = pd.read_csv(f, sep="\t", comment="#")
    frac = t["^18O-_6ns_mpx1"] / (t["^16O-_6ns_mpx1"] + t["^18O-_6ns_mpx1"]) * 100
    sid = os.path.basename(f)[:2]
    ax[0].plot(t["scan"], frac, color=C[i], lw=1.6, label=f"{sid} (명목 {nominal[sid]}%)")
    s4[sid] = round(frac.max(), 1)
    if i == 0:
        for sp, colr in [("^16O-_6ns_mpx1", C[0]), ("WO3-_6ns_mpx1", C[1]), ("Si-_6ns_mpx1", C[2])]:
            ax[1].plot(t["scan"], t[sp].clip(lower=1), color=colr, lw=1.5, label=sp.split("_")[0])
ax[0].set_title("18O 비율 깊이 프로파일 (6 ns)"); ax[0].set_xlabel("scan(깊이 방향 →)")
ax[0].set_ylabel("18O / (16O+18O)  (%)"); ax[0].legend(fontsize=8, frameon=False)
ax[1].set_yscale("log"); ax[1].set_title("S3 시료의 이온별 신호 (WOx막 → Si기판)")
ax[1].set_xlabel("scan"); ax[1].set_ylabel("counts (log)"); ax[1].legend(fontsize=8, frameon=False)
sp = glob.glob(os.path.join(ex, "**", "spectra", "S3*.tsv"), recursive=True)[0]
s = pd.read_csv(sp, sep="\t", comment="#"); s = s[(s.mass_cal_amu > 15.8) & (s.mass_cal_amu < 18.2)]
ax[2].plot(s.mass_cal_amu, s.intensity.clip(lower=.5), color=C[6], lw=.8); ax[2].set_yscale("log")
for mz, nm in [(15.995, "16O-"), (17.003, "OH-"), (17.999, "18O-")]:
    ax[2].axvline(mz, color=GRAY, ls="--", lw=.7); ax[2].text(mz, ax[2].get_ylim()[1] * .5, nm, ha="center", fontsize=8)
ax[2].set_title("S3 질량 스펙트럼 (m/z 16~18)"); ax[2].set_xlabel("m/z (u)"); ax[2].set_ylabel("intensity (log)")
save(fig, "fig4_tofsims.png")
STATS["4"] = dict(samples=4, profiles=len(glob.glob(os.path.join(ex, "**", "depth_profiles", "*.tsv"), recursive=True)),
                  scans=len(t), peak_18O=s4)

# ======================================================================
# 5. 웨이퍼맵 : MixedWM38 (8종 기본 결함, 혼합 38종)
# ======================================================================
z = np.load(glob.glob(os.path.join(folder("5"), "*.npz"))[0])
X5, Y5 = z["arr_0"], z["arr_1"]
names = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "Scratch", "Random"]
from matplotlib.colors import ListedColormap
cm5 = ListedColormap(["white", "#d9e6f5", "#1f4e99", "#e34948"])
fig = plt.figure(figsize=(16, 5.2))
picks = [("Normal", np.where(Y5.sum(1) == 0)[0][0])]
for j, nm in enumerate(names):
    onehot = np.zeros(8, int); onehot[j] = 1
    picks.append((nm, np.where((Y5 == onehot).all(1))[0][0]))
mix, seen = [], set()
for nd in (2, 3, 4):                                  # 결함 2·3·4종 혼합 예시 1개씩
    for i in range(len(Y5)):
        if Y5[i].sum() == nd and tuple(Y5[i]) not in seen:
            mix.append(i); seen.add(tuple(Y5[i])); break
picks += [(" + ".join(names[k] for k in np.where(Y5[i])[0]), i) for i in mix]
for k, (nm, idx) in enumerate(picks):
    a = fig.add_subplot(2, 6, k + 1); a.imshow(X5[idx], cmap=cm5, vmin=0, vmax=3, interpolation="nearest")
    a.set_title(nm, fontsize=8 if "+" in nm else 10); a.axis("off")
save(fig, "fig5_wafermap.png")
combo = pd.read_csv(glob.glob(os.path.join(folder("5"), "label_combinations.csv"))[0], encoding="utf-8-sig")
STATS["5"] = dict(sample=len(X5), size=list(X5.shape[1:]), combos=len(combo),
                  full=int(combo.count_in_full_dataset.sum()), pixel_vals=np.unique(X5).tolist(),
                  n_defect_types=pd.Series(Y5.sum(1)).value_counts().sort_index().to_dict())

# ======================================================================
# 6. SEM 이미지 : Infineon Carinthia 결함 분류
# ======================================================================
p6 = folder("6")
sl = pd.read_csv(os.path.join(p6, "sample_labels.csv"), sep=";", encoding="utf-8-sig")
fl = pd.read_csv(os.path.join(p6, "carinthia_full_label_table.csv"), sep=";", encoding="utf-8-sig")
fig = plt.figure(figsize=(16, 5.6))
gs = fig.add_gridspec(3, 8, width_ratios=[1] * 6 + [.15, 2.2])
for i, (lb, g) in enumerate(sl.groupby("label")):
    for j, fn in enumerate(g.file_name[:3]):
        a = fig.add_subplot(gs[j, i]); a.imshow(Image.open(os.path.join(p6, "images", fn)), cmap="gray"); a.axis("off")
        if j == 0: a.set_title(f"label {lb}", fontsize=10)
a = fig.add_subplot(gs[:, 7]); vc = fl.label.value_counts().sort_index()
a.bar(vc.index.astype(str), vc.values, color=C[0], width=.6); a.set_yscale("log")
for x, v in zip(vc.index.astype(str), vc.values): a.text(x, v * 1.15, f"{v:,}", ha="center", fontsize=8)
a.set_title(f"전체 {len(fl):,}장 라벨 분포 (log)"); a.set_xlabel("label"); a.set_ylabel("이미지 수")
save(fig, "fig6_sem.png")
STATS["6"] = dict(full=len(fl), sample=len(sl), dist=vc.to_dict(), size="480×480 흑백",
                  imbalance=round(vc.max() / vc.min(), 0))

# ======================================================================
# 7. 초분광 큐브 : 4H-SiC (VNIR+SWIR / MWIR / LWIR)
# ======================================================================
p7 = folder("7")
def read_envi(stem):
    h = open(os.path.join(p7, stem + ".hdr"), encoding="utf-8", errors="ignore").read()
    g = lambda k: int(re.search(rf"{k}\s*=\s*(\d+)", h).group(1))
    S, L, B = g("samples"), g("lines"), g("bands")
    wl_m = re.search(r"wavelength\s*=\s*\{([^}]*)\}", h) or re.search(r"band names\s*=\s*\{([^}]*)\}", h)
    wl = np.array([float(v) for v in wl_m.group(1).replace("\n", "").split(",") if v.strip()])
    cube = np.fromfile(os.path.join(p7, stem + ".dat"), dtype="<f4").reshape(B, L, S)   # BSQ: 밴드 × 행 × 열
    return cube, wl
fig, ax = plt.subplots(1, 4, figsize=(17, 4), gridspec_kw={"width_ratios": [1, 1, 1, 2.2]})
s7 = {}
for k, (stem, nm) in enumerate([("4H-SiC_VNIR+SWIR", "VNIR+SWIR"), ("4H-SiC_MWIR", "MWIR"), ("4H-SiC_LWIR", "LWIR")]):
    cube, wl = read_envi(stem)
    wl_um = wl / 1000 if wl.max() > 100 else wl
    if nm == "LWIR" and wl_um.max() < 1: wl_um = 1e4 / wl                 # 파수(cm⁻¹)인 경우 μm로 변환
    mid = cube.shape[0] // 2
    ax[k].imshow(cube[mid], cmap="magma"); ax[k].axis("off")
    ax[k].set_title(f"{nm}  {wl_um[mid]:.2f} μm 밴드\n{cube.shape[2]}×{cube.shape[1]} px, {cube.shape[0]} 밴드", fontsize=9)
    spec = np.nanmean(cube.reshape(cube.shape[0], -1), axis=1)
    o = np.argsort(wl_um); ax[3].plot(wl_um[o], spec[o], color=C[k], lw=1.6, label=nm)
    s7[nm] = dict(shape=list(cube.shape), wl=[round(float(wl_um.min()), 2), round(float(wl_um.max()), 2)])
ax[3].set_xscale("log"); ax[3].set_title("영역별 평균 반사율 스펙트럼"); ax[3].set_xlabel("파장 (μm, log)")
ax[3].set_ylabel("평균 반사율"); ax[3].legend(fontsize=8, frameon=False)
save(fig, "fig7_hsi.png")
STATS["7"] = s7

# ======================================================================
# 8. 원자 구조 : SAIT MLFF (HfO · SiN)
# ======================================================================
p8 = folder("8")
s8 = pd.read_csv(os.path.join(p8, "sait_mlff_structures_summary.csv"))
fig = plt.figure(figsize=(16, 4.4))
for k, mat in enumerate(["HfO", "SiN"]):
    a = fig.add_subplot(1, 3, k + 1)
    for j, sp in enumerate(["Trainset", "OOD"]):
        v = s8[(s8.material == mat) & (s8.split == sp)].energy_per_atom_eV
        a.hist(v, bins=60, color=C[0] if sp == "Trainset" else C[1], alpha=.6, density=True, label=f"{sp} (n={len(v):,})")
    a.set_title(f"{mat}: 원자당 에너지 분포"); a.set_xlabel("energy per atom (eV)"); a.set_ylabel("밀도")
    a.legend(fontsize=8, frameon=False)
lines = open(os.path.join(p8, "SiN_OOD_first30frames.xyz")).read().splitlines()
n = int(lines[0]); atoms = [l.split() for l in lines[2:2 + n]]
el = np.array([a_[0] for a_ in atoms]); xyz = np.array([[float(v) for v in a_[1:4]] for a_ in atoms])
frc = np.linalg.norm(np.array([[float(v) for v in a_[4:7]] for a_ in atoms]), axis=1)
a = fig.add_subplot(1, 3, 3, projection="3d")
for j, e in enumerate(np.unique(el)):
    mk = el == e; a.scatter(*xyz[mk].T, s=40 if e == "Si" else 22, color=C[j], edgecolor="white", lw=.3, label=f"{e} ({mk.sum()})")
a.set_title(f"SiN OOD 1번 프레임 ({n} 원자)"); a.legend(fontsize=8, frameon=False)
a.set_xlabel("x (Å)"); a.set_ylabel("y (Å)"); a.set_zlabel("z (Å)")
save(fig, "fig8_atoms.png")
STATS["8"] = dict(rows=len(s8), by=s8.groupby(["material", "split"]).size().to_dict(),
                  frame_atoms=n, elements={e: int((el == e).sum()) for e in np.unique(el)},
                  max_force=round(frc.max(), 3))
STATS["8"]["by"] = {f"{a_}/{b_}": v for (a_, b_), v in STATS["8"]["by"].items()}

# ======================================================================
# 9. 측정값이 아닌 자료 : ALD 문헌 DB (LLM 추출값 일치 여부)
# ======================================================================
d9 = pd.read_csv(glob.glob(os.path.join(folder("9"), "agreement_flags.csv"))[0])
ct = d9.groupby("field").agreement.value_counts(normalize=True).unstack().fillna(0) * 100
ct = ct.sort_values("agree")
kr = {"material": "물질", "precursor": "전구체", "co_reactant": "반응물", "deposition_temperature": "증착 온도",
      "pulse_time": "펄스 시간", "purge_time": "퍼지 시간", "growth_per_cycle": "사이클당 성장", "chamber_pressure": "챔버 압력"}
fig, ax = plt.subplots(1, 2, figsize=(15, 4.2), gridspec_kw={"width_ratios": [1.6, 1]})
left = np.zeros(len(ct))
for col, colr, lbl in [("agree", C[0], "일치"), ("disagree", C[1], "불일치"), ("one_only", GRAY, "한쪽만 추출")]:
    ax[0].barh([f"{kr[i]} ({i})" for i in ct.index], ct[col], left=left, color=colr, height=.6,
               edgecolor="white", lw=2, label=lbl)
    left += ct[col].values
for i, v in enumerate(ct.agree): ax[0].text(v - 1, i, f"{v:.0f}%", va="center", ha="right", color="white", fontsize=8)
ax[0].set_xlim(0, 100); ax[0].set_xlabel("비율 (%)"); ax[0].set_title("항목별 두 LLM 추출값 일치율")
ax[0].legend(ncol=3, fontsize=8, frameon=False, loc="lower right", bbox_to_anchor=(1, 1.06))
vc9 = d9.groupby("doi").size().value_counts().sort_index()
ax[1].bar(vc9.index.astype(str), vc9.values, color=C[0], width=.6)
ax[1].set_title("논문(DOI)당 추출된 항목 수"); ax[1].set_xlabel("항목 수"); ax[1].set_ylabel("논문 수")
save(fig, "fig9_ald.png")
ex9 = d9[d9.agreement == "disagree"].groupby("field").head(1)[["field", "value_production", "value_gemma"]].head(5)
STATS["9"] = dict(rows=len(d9), dois=d9.doi.nunique(), agree=round((d9.agreement == "agree").mean() * 100, 1),
                  by_field={k: round(v, 1) for k, v in ct.agree.items()},
                  examples=ex9.values.tolist())

json.dump(STATS, open(os.path.join(OUT, "stats.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print("완료 →", OUT)
