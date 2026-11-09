# HW01 執行說明

## 1. 環境

建議使用 Python 3.9–3.12。於專案根目錄執行：

```bash
python -m venv .venv
```

Windows：

```bash
.venv\Scripts\activate
```

macOS / Linux：

```bash
source .venv/bin/activate
```

安裝套件：

```bash
python -m pip install -r hw01/requirements.txt
```

## 2. 執行實驗

在 Repo 根目錄執行：

```bash
python hw01/code/run_experiment.py
```

若已經切換到 `hw01/` 目錄：

```bash
python code/run_experiment.py
```

程式固定 `random_state=0`，使用 `load_digits()`，以 70%/30% 並採用 stratify 切分。模型參數只使用訓練集估計，測試集只用於評估。

## 3. 輸出

- `figures/`：三張混淆矩陣、QDA 正則化掃描圖、二維決策區域圖。
- `results/accuracy.csv`：三個主要模型的測試準確率。
- `results/qda_regularization.csv`：QDA 正則化掃描數據。
- `results/summary.json`：實驗設定、準確率與 GaussianNB 比對數據。

`results/` 會在第一次執行時自動建立。請將程式輸出的數字與圖表整理到 `report.md`，不要自行捏造數據。

## 4. 實作說明

`gaussian_nb.py` 使用 NumPy 計算每一類的先驗、平均數與各特徵變異數，並在對數空間計算後驗機率。沒有使用 `sklearn.naive_bayes` 進行自製模型的訓練或預測。
