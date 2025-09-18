````markdown
# 📊 Financial Report & Portfolio Simulator

Questo progetto è un **dashboard interattivo in Streamlit** per:
1. Analizzare **entrate e uscite personali** da un file Excel/CSV.
2. Calcolare il **saldo annuale**.
3. Simulare la costruzione di un **portafoglio di investimenti** con metriche di rischio/rendimento.
4. Visualizzare la **crescita nel tempo dell’investimento** tramite capitalizzazione composta.
5. (Avanzato) Mostrare **Frontiera Efficiente** e **Contribution Analysis**.

---

## 🚀 Funzionalità principali

### 📒 Analisi finanziaria personale
- Upload di un file Excel/CSV con entrate e uscite.
- Report mensile e annuale:
  - Entrate totali / Uscite totali
  - Saldo mensile e annuale
- Grafici interattivi:
  - Entrate vs Uscite per mese
  - Distribuzione per tipologia
  - Istogramma saldo mensile
- Esportazione template Excel/CSV pronto all’uso.

### 💹 Simulazione Portafoglio
- Selezione **Asset Class** predefinite (Azioni, Bond, Crypto).
- Possibilità di inserire ticker manuali.
- Assegnazione pesi ai titoli.
- Metriche del portafoglio:
  - ✅ Rendimento atteso annuo
  - ✅ Volatilità annua
  - ✅ Sharpe Ratio
  - ✅ Sortino Ratio
  - ✅ Value at Risk (VaR) 95%
  - ✅ Expected Shortfall (ES) 95%
  - ✅ Max Drawdown
- Grafici interattivi:
  - Andamento cumulato
  - Distribuzione dei rendimenti
  - Allocazione pesi
  - Rolling volatility
  - Drawdown
  - Heatmap correlazioni
  - Contribution Analysis (rendimento & rischio)

### 📈 Simulazione crescita capitale con metodo Copula (t-student)
- Usa il **saldo annuale** come capitale investibile.
- Simula crescita su orizzonte da 1 a 30 anni.
- Considera **capitalizzazione composta**.
- Grafico stacked: quota investita + rendimento maturato.

### 🧮 Analisi avanzata
- **Efficient Frontier**:  
  - Simulazione di migliaia di portafogli casuali.  
  - Scatterplot rischio/rendimento.  
  - Evidenziazione portafoglio selezionato.  
- **Contribution Analysis**:  
  - Breakdown del contributo di ogni asset a rendimento e rischio.  

---

## 📂 Struttura del progetto

```bash
.
├── app.py                   # File principale Streamlit
├── utils/
│   ├── portfolio_utils.py    # Funzioni di analisi portafoglio
│   └── ...
├── requirements.txt         # Dipendenze del progetto
└── README.md                # Documentazione
````

---

## ⚙️ Installazione

1. Clona il repository:

   ```bash
   git clone https://github.com/tuo-username/financial-report.git
   cd financial-report
   ```

2. Crea un ambiente virtuale e installa le dipendenze:

   ```bash
   python -m venv venv
   source venv/bin/activate   # macOS/Linux
   venv\Scripts\activate      # Windows

   pip install -r requirements.txt
   ```

3. Avvia l’app:

   ```bash
   streamlit run app.py
   ```

---

## 📊 Esempio file Excel

Formato richiesto:

| Tipo    | Tipologia | Dettaglio    | gen  | feb  | mar  | ... | dic  |
| ------- | --------- | ------------ | ---- | ---- | ---- | --- | ---- |
| Entrate | Stipendio | Azienda XYZ  | 1000 | 1000 | 1000 | ... | 1000 |
| Uscite  | Affitto   | Casa Milano  | 600  | 600  | 600  | ... | 600  |
| Uscite  | Spesa     | Supermercato | 300  | 250  | 280  | ... | 310  |

⚠️ Colonne richieste:

* **Tipo**: Entrate / Uscite
* **Tipologia**: Categoria
* **Dettaglio**: Descrizione
* **Mesi (gen-dic)**: valori numerici o in formato `€`

---

## 📦 Dipendenze principali

* [Streamlit](https://streamlit.io/)
* [Pandas](https://pandas.pydata.org/)
* [Plotly](https://plotly.com/python/)
* [NumPy](https://numpy.org/)
* [yfinance](https://pypi.org/project/yfinance/) (download dati storici)
* [openpyxl](https://openpyxl.readthedocs.io/) (supporto Excel)

---

## 🔮 Possibili sviluppi futuri

* Integrazione con API bancarie (es. PSD2).
* Portfolio optimization (es. **Black-Litterman**).
* Backtest con strategie dinamiche.

---

## 👨‍💻 Autori

* Fabrizio Di Sciorio, PhD
* Giulia Cartei

```

