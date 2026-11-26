"""
Strategie di copertura.

Ogni strategia è un insieme di regole che, giorno per giorno, dicono quante
posizioni tenere. Il backtest (backtest.py) applica le regole, paga i costi di
bid-ask e calcola il P&L.

Regole comuni
-------------
- Le posizioni sono in barili e vengono arrotondate a contratti interi da 1.000 bbl.
- Delta: sulla gamba i si tengono -delta_i barili di futures (delta del collar più
  quello delle eventuali opzioni di copertura su quella gamba).
- Banda di non intervento (delta_band_lots): si tratta solo se la posizione
  desiderata differisce da quella attuale di più di tanti contratti. Con banda 0
  si ribilancia ogni giorno; con una banda si pagano meno spread ma si lascia
  scoperto un piccolo delta.
- Copertura statica (static_hedge = True): all'emissione si vende il "collar
  opposto" con opzioni quotate sul front contract (Apr21): si vendono put con
  strike intorno a Kput e si comprano call con strike intorno a Kcall. Siccome
  gli strike esatti non sono quotati, si usano i due strike quotati più vicini
  con pesi lineari (se K è fuori dagli strike quotati, solo il più vicino).
  Quantità: S3 usa 1M bbl per lato; S4 (static_scale = "collar_vega") quanto basta perché la
  vega delle put vendute sia uguale alla vega della put del collar, e lo stesso
  per le call. Il collar è sulla media, che è meno volatile del prezzo di un
  giorno solo: ha meno vega di un'opzione europea a parità di nozionale, quindi
  1M bbl di opzioni europee coprirebbero troppa vega; il resto lo regola l'opzione dinamica.
  Senza opzione dinamica (S3) conviene invece tenere 1M bbl. La posizione si costruisce in
  static_phase_days giorni (comprarla tutta il primo giorno costerebbe molti
  spread in un giorno solo) e si chiude quando alle opzioni mancano meno di
  static_close_days giorni alla scadenza, tranne le opzioni che valgono meno del loro
  bid-ask (static_let_expire): chiuderle costerebbe più di quanto valgono, quindi si
  lasciano scadere e si incassa o paga il valore intrinseco. Toglie la maggior parte del rischio,
  compresa la forma dello smile; il resto lo coprono futures e opzione dinamica.
- Delta delle opzioni (option_delta_smile = True): calcolato riprezzando l'opzione
  con lo smile che si sposta insieme al futures, come per il collar. La delta di
  Black-76 pura ignora lo skew e, nel 2020, sbaglia molto la copertura.
- Vega: si neutralizza la vega del collar con opzioni quotate sul secondo futures
  (May21 nel training, le cui opzioni scadono dopo la fine del collar).
  vega_mode = "single": un'opzione sola contro tutta la vega residua, con strike
    il più vicino possibile a Kput oppure (vega_reference = "dominant") allo
    strike tra Kput e Kcall che in quel momento pesa di più sulla vega.
  Un'opzione si cambia solo se smette di essere negoziabile o se ne esiste una
  più vicina allo strike di riferimento di almeno option_switch_distance; la quantità si aggiorna solo se
  si discosta da quella desiderata di più di vega_band (in proporzione), per non
  pagare spread ogni giorno. Quando si sceglie un'opzione nuova si prende la OTM.
"""
from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class HedgeStrategy:
    name: str                       # nome breve (file e tabelle)
    label: str                      # descrizione leggibile
    hedge_delta: bool = True
    delta_band_lots: int = 0        # banda di non intervento sui futures, in contratti
    vega_mode: str = "none"         # "none" o "single"
    vega_band: float = 0.05         # si ribilancia l'opzione se |ψ* - ψ| > vega_band·|ψ*|
    option_max_rel_spread: float = 0.60   # opzione negoziabile solo se (ask-bid)/mid <= soglia
    option_min_ttm_days: int = 20         # e se mancano almeno tanti giorni alla sua scadenza
    option_switch_distance: float = 10.0  # si cambia opzione se ce n'è una più vicina al riferimento di almeno tanto ($)
    vega_reference: str = "kput"          # strike di riferimento dell'opzione dinamica: "kput" o "dominant"
    # copertura statica: all'emissione si vende il "collar opposto" con opzioni quotate
    static_hedge: bool = False
    static_leg: int = 0                   # gamba delle opzioni statiche (0 = front contract, es. Apr21)
    static_let_expire: bool = True        # le opzioni statiche con time value <= spread si lasciano scadere
    static_close_days: int = 10           # si chiudono quando mancano meno giorni alla loro scadenza
    static_exit_days: int = 1             # giorni su cui distribuire l'eventuale chiusura anticipata (>1 = uscita graduale)
    static_trim_days: int = 0             # se >0, negli ultimi N giorni riduce le ali statiche sovra-coperte invece di aprire calendar spread
    static_phase_days: int = 1            # giorni di costruzione statica (0 = adattivo k* in funzione di spread e vol ATM)
    static_defer_dynamic: bool = False    # durante la costruzione statica, evita di aprire/chiudere l'opzione dinamica
    static_scale: str = "notional"        # notional = 1M bbl per gamba; collar_vega = stessa vega della gamba del collar
    static_vega_mult: float = 1.0         # con collar_vega: frazione della vega del collar coperta staticamente
    option_delta_smile: bool = True       # delta delle opzioni coerente con lo smile (sticky moneyness)

    @property
    def vega_hedge(self) -> bool:
        return self.vega_mode != "none"


UNHEDGED = HedgeStrategy("S0_nessuna", "Nessuna copertura", hedge_delta=False)
DELTA_DAILY = HedgeStrategy("S1_delta", "Delta con futures")
DELTA_VEGA = HedgeStrategy("S2_delta_vega", "Delta + vega con un'opzione",
                           vega_mode="single", vega_reference="dominant")
STATIC_DELTA = HedgeStrategy("S3_statica_delta", "Collar opposto statico + delta",
                             static_hedge=True, static_phase_days=10)   # senza opzione dinamica: 1M bbl
STATIC_DELTA_VEGA = HedgeStrategy("S4_statica_delta_vega", "Collar opposto statico + delta + vega",
                                  static_hedge=True, static_phase_days=0, static_defer_dynamic=True,
                                  static_trim_days=75, static_exit_days=5,
                                  static_scale="collar_vega", vega_mode="single", vega_reference="dominant")
# Costruzione adattiva k* (spread vs vol ATM) + rinvio della dinamica durante la costruzione +
# riduzione graduale delle ali statiche sovra-coperte negli ultimi 75 giorni + uscita graduale su 5 giorni.

MAIN_STRATEGIES = [UNHEDGED, DELTA_DAILY, DELTA_VEGA, STATIC_DELTA, STATIC_DELTA_VEGA]

# Strategia consigliata (la migliore sul training e sulla prova generale)
BEST = STATIC_DELTA_VEGA

# Varianti provate e scartate (per il report): ognuna cambia una sola cosa rispetto a una
# delle strategie principali
VARIANTS = {
    "S1 con banda di 10 contratti sui futures": replace(DELTA_DAILY, delta_band_lots=10),
    "S2 con delta delle opzioni di Black-76 (senza smile)": replace(DELTA_VEGA, option_delta_smile=False),
    "S4 con delta delle opzioni di Black-76 (senza smile)": replace(STATIC_DELTA_VEGA, option_delta_smile=False),
    "S4 con collar statico sul May21 invece che sull'Apr21": replace(STATIC_DELTA_VEGA, static_leg=1),
    "S4 con collar statico da 1M bbl (senza riscalare sulla vega)": replace(STATIC_DELTA_VEGA, static_scale="notional"),
}
