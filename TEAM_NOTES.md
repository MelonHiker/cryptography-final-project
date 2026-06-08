# 給組員：`experiments-and-eval` 分支說明

這條分支從 `attack_simulation` 拉出來，**沒有改動 `main` 和 `attack_simulation`**。

## 跟 `attack_simulation` 的差別（我們新增/修改了什麼）

| 檔案 | 變更 | 說明 |
|---|---|---|
| `experiments/` (新) | 離線實驗腳本 | 消融比較（只聲學 / 只時序 / 融合）+ ROC 曲線 + 分數直方圖 + EER/AUC/FAR/FRR，可吃合成或真實 CSV |
| `tests/` (新) | 15 個單元測試 | 驗證 `features.py`(46 維、MFCC) 與 `modeling.py`(訓練/存讀/門檻) |
| `collect.py` (新) | 平易近人蒐集介面 | 單視窗：填名字→按開始→打通關密語→存 `data/<名字>.csv`，給同學幫忙蒐 imposter 資料用 |
| `smoke_test.py` (新) | 免硬體冒煙測試 | 只需 numpy+sklearn 就能驗證核心管線 |
| `keystroke_auth/modeling.py` | **新增 `strict` 開關** | `evaluate_with_artifacts(..., strict=False)`。**預設寬鬆**（單一融合模型），嚴格模式才啟用四道閘門 |
| `keystroke_auth/gui.py` | 登入分頁加開關 + OTP 修正 | 嚴格/寬鬆勾選框；OTP 收件信箱改在 Settings 固定、登入畫面唯讀 |
| `keystroke_auth/config.py` | 新增 `strict_mode` 欄位 | 預設 `False` |
| `attack_simulator/simulation.py` | 攻擊模擬固定 `strict=True` | 保留你原本的攻擊報告數字不變 |
| `README.md` | Windows 設定、檔名修正 | — |

## 重點：為什麼改「太嚴格」
四道閘門用 AND 邏輯串接，誤拒本人的機率相乘。實測 **嚴格模式誤拒本人 FRR≈20%，寬鬆模式≈0%**。
所以正式登入**預設用寬鬆**；嚴格模式保留成開關，報告當「更高安全性的選項」來談。

## 下一步可以一起做
1. 用 `collect.py` 找 3~5 位同學各蒐 ~15 筆真實 imposter 資料。
2. 跑 `python -m experiments.run_experiments --owner-csv <你的>.csv --imposter-csv <同學>.csv`，把真實的 ROC / FAR / FRR 圖表放進報告。
3. 決定報告主軸：寬鬆模型的可用性數據 + 嚴格模型的安全性數據（攻擊模擬）兩邊都呈現。

---

### 一句話通知版（貼群組用）
> 我從 `attack_simulation` 開了新分支 `experiments-and-eval`：加了離線實驗腳本（ROC/消融/EER）、15 個單元測試、一個給同學用的簡易蒐集介面，並把「太嚴格」的模型改成可切換（預設寬鬆，FRR 從 ~20% 降到 ~0%），也修了 OTP 收件人漏洞。下一步可以一起蒐真實資料跑出報告圖表。
