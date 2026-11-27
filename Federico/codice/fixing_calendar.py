"""
Calendario: scadenze ICE Brent, giorni di fixing e regola di roll (Annex B).

Nome del file: "fixing_calendar" e non "calendar", perché "calendar" è un
modulo della libreria standard di Python e lo nasconderemmo.

Regole usate
------------
- Codici contratto stile Bloomberg: "CO" + lettera del mese + anno a 2 cifre,
  es. COJ21 = consegna aprile 2021.
- Scadenza del futures ICE Brent: ultimo giorno lavorativo del secondo mese
  precedente il mese di consegna (Apr21 -> 26/2/2021, Dec22 -> 31/10/2022).
- Fixing: tutti i giorni lavorativi del mese di media.
- Roll (Annex B): il front contract fissa "fino al penultimo giorno di fixing
  prima della scadenza, incluso"; dal giorno di fixing successivo si usa il
  next contract. Interpretazione confermata dai prof nel Q&A del 10/11: il giorno
  di scadenza conta come ultimo giorno di fixing, quindi il "penultimo" è il
  giorno di fixing precedente la scadenza. Feb 2021: Apr21 scade il 26/2 ->
  Apr21 dall'1 al 25/2 (19 fixing), May21 solo il 26/2 (1 fixing).
  Ott 2022: Dec22 scade il 31/10 -> Dec22 fino al 28/10, Jan23 il 31/10.
  La scadenza di riferimento dipende da cfg.roll_rule ("futures_expiry", quella
  ufficiale, o "option_expiry", usata solo nella prova generale su dic 2020).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import CollarConfig

MONTH_CODES = {"F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6,
               "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12}
CODE_OF_MONTH = {m: c for c, m in MONTH_CODES.items()}


# ----------------------------------------------------------------- contratti
def parse_contract(code: str) -> tuple[int, int]:
    """'COJ21' -> (2021, 4): anno e mese di consegna."""
    if len(code) != 5 or not code.startswith("CO") or code[2] not in MONTH_CODES:
        raise ValueError(f"Codice contratto non riconosciuto: {code}")
    return 2000 + int(code[3:]), MONTH_CODES[code[2]]


def contract_code(year: int, month: int) -> str:
    """(2021, 4) -> 'COJ21'."""
    return f"CO{CODE_OF_MONTH[month]}{year % 100:02d}"


def business_days(start, end, holidays=()) -> pd.DatetimeIndex:
    """Giorni lavorativi (lun-ven) tra start ed end inclusi, meno le festività."""
    return pd.bdate_range(start, end, freq="C", holidays=list(holidays))


def futures_expiry(code: str, holidays=()) -> pd.Timestamp:
    """Scadenza ICE Brent: ultimo giorno lavorativo del secondo mese prima della consegna."""
    year, month = parse_contract(code)
    ref = pd.Timestamp(year=year, month=month, day=1) - pd.DateOffset(months=2)
    days = business_days(ref, ref + pd.offsets.MonthEnd(0), holidays)
    return days[-1]


def front_and_next(year: int, month: int, holidays=()) -> tuple[str, str]:
    """Contratti di riferimento per un mese di media.

    Il front è il contratto che scade dentro il mese di media (per il Brent ce
    n'è sempre esattamente uno: consegna due mesi dopo). Il next è il successivo.
    """
    first = pd.Timestamp(year=year, month=month, day=1)
    d1 = first + pd.DateOffset(months=2)
    d2 = first + pd.DateOffset(months=3)
    return contract_code(d1.year, d1.month), contract_code(d2.year, d2.month)


# ------------------------------------------------------------------- fixing
@dataclass
class FixingSchedule:
    table: pd.DataFrame          # colonne: fixing_date, contract, leg (1 o 2)
    roll_reference: pd.Timestamp # scadenza usata per il roll
    roll_rule: str

    @property
    def dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.table.fixing_date)

    @property
    def n1(self) -> int:
        return int((self.table.leg == 1).sum())

    @property
    def n2(self) -> int:
        return int((self.table.leg == 2).sum())

    @property
    def N(self) -> int:
        return len(self.table)

    @property
    def weights(self) -> tuple[float, float]:
        return self.n1 / self.N, self.n2 / self.N

    @property
    def last_fixing(self) -> pd.Timestamp:
        return self.table.fixing_date.iloc[-1]

    def year_fractions(self, valuation_date) -> np.ndarray:
        """θ_k = (t_k - t) / 365 per ogni fixing (ACT/365, come T delle opzioni)."""
        t = pd.Timestamp(valuation_date)
        return ((self.dates - t).days.values / 365.0).astype(float)

    def describe(self) -> str:
        t = self.table
        l1, l2 = t[t.leg == 1], t[t.leg == 2]
        w1, w2 = self.weights
        return (
            f"Regola di roll: {self.roll_rule} (scadenza di riferimento {self.roll_reference.date()})\n"
            f"  {l1.contract.iloc[0]}: {self.n1} fixing, dal {l1.fixing_date.iloc[0].date()} al {l1.fixing_date.iloc[-1].date()}\n"
            f"  {l2.contract.iloc[0]}: {self.n2} fixing, dal {l2.fixing_date.iloc[0].date()} al {l2.fixing_date.iloc[-1].date()}\n"
            f"  pesi w1 = {w1:.3f}, w2 = {w2:.3f}"
        )


def build_fixing_schedule(cfg: CollarConfig, option_expiry_front: pd.Timestamp | None = None) -> FixingSchedule:
    """Calcola i giorni di fixing e assegna a ognuno il contratto di riferimento.

    option_expiry_front serve solo con roll_rule = "option_expiry"
    (si prende dai dati: MarketData.option_expiry(cfg.front_contract)).
    """
    front, nxt = cfg.legs
    expected = front_and_next(cfg.averaging_year, cfg.averaging_month)
    if (front, nxt) != expected:
        raise ValueError(f"Le gambe in config {front, nxt} non corrispondono al mese di media: attese {expected}")

    first = pd.Timestamp(year=cfg.averaging_year, month=cfg.averaging_month, day=1)
    fixings = business_days(first, first + pd.offsets.MonthEnd(0), cfg.holidays)

    if cfg.roll_rule == "futures_expiry":
        ref = futures_expiry(front, cfg.holidays)
    else:
        if option_expiry_front is None:
            raise ValueError("Con roll_rule='option_expiry' serve la scadenza delle opzioni del front contract")
        ref = pd.Timestamp(option_expiry_front)

    upto = fixings[fixings <= ref]
    if len(upto) < 2:
        raise ValueError("Meno di due giorni di fixing entro la scadenza: regola di roll non applicabile")
    last_front = upto[-2]  # penultimo giorno di fixing entro la scadenza (Q&A del 10/11)

    table = pd.DataFrame({"fixing_date": fixings})
    table["leg"] = np.where(table.fixing_date <= last_front, 1, 2)
    table["contract"] = np.where(table.leg == 1, front, nxt)
    return FixingSchedule(table=table, roll_reference=ref, roll_rule=cfg.roll_rule)


def payment_date(cfg: CollarConfig, schedule: FixingSchedule) -> pd.Timestamp:
    """Data di pagamento del collar: ultimo fixing + eventuale ritardo (IPOTESI in config)."""
    if cfg.payment_lag_bdays == 0:
        return schedule.last_fixing
    return business_days(schedule.last_fixing, schedule.last_fixing + pd.Timedelta(days=30),
                         cfg.holidays)[cfg.payment_lag_bdays]


def year_frac(start, end, weekend_weight: float = 1.0):
    """Anni tra start (oggi, dopo la chiusura) ed end, vettoriale.

    weekend_weight = 1: giorni di calendario / 365 (ACT/365).
    weekend_weight = 0: solo giorni di borsa (/252): nel weekend il prezzo non si muove,
    quindi il tempo "di volatilità" non scorre.
    Valori intermedi pesano sabato e domenica come frazione di un giorno di borsa.
    """
    s = np.asarray(pd.to_datetime(start), dtype="datetime64[D]")
    e = np.asarray(pd.to_datetime(end), dtype="datetime64[D]")
    cal = (e - s).astype("timedelta64[D]").astype(float)
    if weekend_weight >= 1.0:
        out = cal / 365.0
    else:
        one = np.timedelta64(1, "D")
        bus = np.busday_count(s + one, e + one).astype(float)
        out = (bus + weekend_weight * (cal - bus)) / (252.0 + weekend_weight * 113.0)
    return float(out) if np.ndim(out) == 0 else out
