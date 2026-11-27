"""
Configurazione del progetto Hedging Competition.

Tutto ciò che cambia tra il collar di training (feb 2021) e quello di test
(ott 2022) sta qui. Il resto del codice legge solo da un oggetto CollarConfig,
quindi per il test basta usare TEST invece di TRAIN.

Le ipotesi non confermate dai prof sono parametri espliciti, segnati con "IPOTESI".
Le scelte di modello (smile, correlazione, Monte Carlo) stanno in ModelConfig;
le strategie di copertura sono definite in strategy.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

# Cartella "Hedging Competition" (questo file sta in Hedging Competition/codice/)
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dati"
OUTPUT_DIR = ROOT / "output"


FUTURES_FILE = DATA_DIR / "COJ21_COZ22_historical_series.xlsx"


@dataclass(frozen=True)
class IVFilters:
    """Filtri per decidere quali opzioni usare per le volatilità implicite.

    Le soglie sono relative (non in dollari fissi), così valgono anche nel 2022.
    max_rel_spread e min_price riprendono le soglie dell'esempio del prof
    (ExampleHedging.m: tresh_bid_ask = 0.6, tresh_penny = 0.1).
    """
    otm_only: bool = True          # solo put con K < F e call con K >= F
    min_price: float = 0.10        # mid minimo ($/bbl): sotto, il prezzo è quasi tutto "rumore di tick"
    max_rel_spread: float = 0.60   # (ask - bid) / mid massimo
    min_ttm_days: int = 5          # niente opzioni a pochi giorni dalla scadenza
    min_iv: float = 0.01           # vol implicite fuori da questo intervallo sono considerate errori
    max_iv: float = 3.00


@dataclass(frozen=True)
class ModelConfig:
    """Scelte di modello per smile, correlazione e Monte Carlo."""
    # smile in log-moneyness x = ln(K/F): "pchip" (interpolazione, scelta principale)
    # oppure "quadratic" (parabola ai minimi quadrati, per confronto); fuori dagli
    # strike quotati la pendenza al bordo viene smorzata con tanh (wing_decay > 0,
    # C¹ continua: evita salti di delta sul bordo; 0.0 = estrapolazione piatta netta)
    smile_method: str = "pchip"
    smile_min_points: int = 3       # con meno punti si usa lo smile dell'altra gamba
    wing_decay: float = 0.05
    # prezzo da cui si calibra lo smile: "settle" (gli stessi prezzi con cui si marcano ogni giorno
    # le opzioni di copertura) oppure "mid" (media bid-ask)
    smile_price: str = "settle"
    # tempo di volatilità: 0 = solo giorni di borsa (/252, nel weekend la varianza non si accumula),
    # 1 = giorni di calendario (/365); valori intermedi pesano il weekend come frazione di giorno
    weekend_weight: float = 0.0        # raggio (in log-moneyness) di smorzamento C¹ sulle ali
    # correlazione: media mobile esponenziale (EWMA) dei log-rendimenti dei due futures
    ewma_lambda: float = 0.94
    rho_prior: float = 0.99         # valore usato finché ci sono meno di corr_min_obs rendimenti
    corr_min_obs: int = 10
    # Monte Carlo (solo per validare il moment matching)
    mc_paths: int = 200_000
    mc_seed: int = 42


@dataclass(frozen=True)
class CollarConfig:
    name: str
    data_file: Path                      # dataset delle opzioni (parquet)
    futures_file: Path                   # serie storiche dei futures (Excel, un foglio per contratto)
    futures_sheet: str                   # foglio da usare per questo collar (es. "COJ21")

    # --- Prodotto (term sheet, Annex A/B) -------------------------------
    emission_date: pd.Timestamp          # data di emissione: si risolve Kput qui
    backtest_end: pd.Timestamp           # ultimo giorno del backtest
    averaging_year: int                  # anno del mese di media
    averaging_month: int                 # mese di media (2 = febbraio)
    front_contract: str                  # gamba 1, es. "COJ21" (Apr21)
    next_contract: str                   # gamba 2, es. "COK21" (May21)
    k_call: float = 78.0                 # strike della call ($/bbl)
    notional_bbl: float = 1_000_000.0    # nozionale in barili
    lot_bbl: float = 1_000.0             # 1 contratto ICE Brent = 1.000 bbl

    # --- IPOTESI in attesa di conferma ------------------------------------
    # "futures_expiry": il roll si conta dalla scadenza del futures (Apr21 -> 26/2/2021):
    #                   Apr21 dall'1 al 25/2, May21 il 26/2 (19/1, confermato nel Q&A del 10/11)
    # "option_expiry":  il roll si conta dalla scadenza delle opzioni; usato solo nella
    #                   prova generale su dic 2020, dove mancano i prezzi dopo quella data
    roll_rule: str = "futures_expiry"
    # giorni lavorativi tra l'ultimo fixing e il pagamento del collar
    payment_lag_bdays: int = 0
    # contratti utilizzabili per la copertura (None = solo le due gambe)
    hedge_contracts: tuple[str, ...] | None = None
    # festività da escludere dai giorni lavorativi (oltre ai weekend)
    holidays: tuple[pd.Timestamp, ...] = field(default_factory=tuple)

    # --- Scelte di modellazione ------------------------------------------------
    iv_filters: IVFilters = field(default_factory=IVFilters)
    model: ModelConfig = field(default_factory=ModelConfig)

    def __post_init__(self) -> None:
        if self.roll_rule not in ("futures_expiry", "option_expiry"):
            raise ValueError(f"roll_rule non valida: {self.roll_rule}")

    @property
    def legs(self) -> tuple[str, str]:
        return (self.front_contract, self.next_contract)

    @property
    def hedge_universe(self) -> tuple[str, ...]:
        return self.hedge_contracts or self.legs


TRAIN = CollarConfig(
    name="train_feb2021",
    data_file=DATA_DIR / "unpivoted_bid_offer_train_clean.parquet",   # versione del 31/10
    futures_file=FUTURES_FILE,
    futures_sheet="COJ21",
    emission_date=pd.Timestamp("2020-01-06"),
    backtest_end=pd.Timestamp("2021-02-28"),
    averaging_year=2021,
    averaging_month=2,
    front_contract="COJ21",   # Apr21
    next_contract="COK21",    # May21
)

# Test out-of-sample: dati rilasciati dopo l'8/12.
# Data di emissione 6/4/2021: confermata nel Q&A del 10/11.
# Il nome del file sarà da aggiornare quando i prof lo pubblicano.
# NB: il foglio COZ22 dell'Excel contiene già i prezzi del 2022. Va usato SOLO
# nell'esecuzione finale sul test (8-11/12), mai per mettere a punto la strategia.
TEST = CollarConfig(
    name="test_oct2022",
    data_file=DATA_DIR / "full_df_test.parquet",
    futures_file=FUTURES_FILE,
    futures_sheet="COZ22",
    emission_date=pd.Timestamp("2021-04-06"),
    backtest_end=pd.Timestamp("2022-10-31"),
    averaging_year=2022,
    averaging_month=10,
    front_contract="COZ22",   # Dec22
    next_contract="COF23",    # Jan23
)

# Prova generale "fuori campione" dentro il training: un collar inventato sulla
# media di dicembre 2020 (futures Feb21/Mar21), con la stessa logica del collar
# vero. Serve solo a controllare che il codice non sia tarato su febbraio 2021.
# Il roll si conta dalla scadenza delle opzioni perché per Feb21 non abbiamo
# i prezzi del futures dopo la scadenza delle sue opzioni (23/12/2020).
# Strike della call scelto con la stessa moneyness del collar vero: 78/61,40 ≈ 1,27.
PSEUDO_OOS = CollarConfig(
    name="prova_dic2020",
    data_file=DATA_DIR / "unpivoted_bid_offer_train_clean.parquet",
    futures_file=FUTURES_FILE,
    futures_sheet="COJ21",
    emission_date=pd.Timestamp("2020-01-06"),
    backtest_end=pd.Timestamp("2020-12-31"),
    averaging_year=2020,
    averaging_month=12,
    front_contract="COG21",   # Feb21
    next_contract="COH21",    # Mar21
    k_call=79.0,
    roll_rule="option_expiry",
    holidays=(pd.Timestamp("2020-12-25"),),
)
