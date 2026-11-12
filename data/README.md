# 데이터 준비

원본 데이터는 라이선스 때문에 저장소에 넣지 않았습니다. 각 배포처에서 받아 아래 폴더 이름대로 `data/` 안에 넣으면 코드가 그대로 돌아갑니다. 코드는 폴더 이름 앞의 번호(`1_`, `2_` …)로 폴더를 찾습니다.

| 폴더 | 필요한 파일 | 출처 · 라이선스 |
|---|---|---|
| `1_웨이퍼단위_요약표_09_Infineon_AlCu_PVD/` | `apc_spc_AlCu_inputs_outputs.csv` | Infineon AlCu PVD APC/SPC 데이터셋 |
| `2_센서_시계열_05_LAM9600/` | `lam9600_machine_timeseries.csv`, `lam9600_wafer_metadata.csv` | LAM 9600 Metal Etch 데이터를 표 형태로 정리한 것 |
| `3_위치별_계측_06_Fraunhofer_Bosch_Etch/` | `Si_Oxide_etch_89_points.csv`, `Lot_status.xlsx` | Sayyed 외, Fraunhofer ENAS / TU Chemnitz, Bosch 플라즈마 식각 멀티모델 데이터셋 |
| `4_스펙트럼_프로파일_12_ToF_SIMS_WOx/` | `exports_WOx.zip` | Ievlev A. V., ORNL, Zenodo DOI 10.5281/zenodo.23037549 · CC BY 4.0 |
| `5_웨이퍼맵_20_MixedWM38_샘플/` | `MixedWM38_sample_190.npz`, `label_combinations.csv` | MixedWM38 혼합 웨이퍼맵 결함 데이터셋 (샘플 190장) |
| `6_SEM_이미지_16_Infineon_Carinthia_샘플/` | `sample_labels.csv`, `carinthia_full_label_table.csv`, `images/` | Infineon Carinthia SEM 결함 이미지 데이터셋 (샘플 50장) |
| `7_초분광_큐브_14_HSI_4H-SiC/` | `4H-SiC_*.hdr`, `4H-SiC_*.dat` | 반도체 시료 초분광(HSI) 반사율 데이터셋 |
| `8_원자_구조_SAIT_SiN_샘플/` | `sait_mlff_structures_summary.csv`, `SiN_OOD_first30frames.xyz` | SAIT MLFF 데이터셋 · CC BY-NC-SA 4.0 (비상업) |
| `9_측정값_아닌_자료_08_ALD_문헌DB/` | `agreement_flags.csv` | ALD 공정 파라미터 DB v2.9 |

> 출처 표기는 각 폴더에 들어 있던 README·라이선스 파일을 기준으로 했습니다. 공개하기 전에 각 배포처의 최신 라이선스와 인용 방식을 확인해 주세요.
