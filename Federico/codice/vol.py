"""
Volatilità implicite delle opzioni su futures (Black-76 non scontato, r = 0).

Perché r = 0: le opzioni ICE Brent hanno margining futures-style, cioè il premio
non si paga all'acquisto ma si regola giorno per giorno come un futures. Non c'è
denaro da finanziare, quindi niente sconto (Asay, 1982). Con questo margining
l'esercizio anticipato non è mai ottimale (Lieu, 1990): l'opzione americana vale
come quella europea e possiamo usare la formula europea.

Contenuto
---------
- black76_price, black76_vega : formula di Black-76 e sua derivata rispetto a σ
- implied_vol                 : inversione numerica (bisezione), vettorizzata
- compute_implied_vols        : vol implicite di tutte le opzioni, con i filtri
- atm_vol                     : vol "at the money" approssimata, giorno per giorno
- Smile, fit_smile            : curva continua della vol in funzione dello strike
- calibrate_smiles            : uno smile per ogni giorno e per ogni gamba

Modello dello smile
-------------------
In ogni giorno e per ogni futures, la vol implicita è una funzione della
log-moneyness x = ln(K / F), costruita sulle opzioni che superano i filtri:
- metodo principale "pchip": interpolazione monotona a tratti che passa per i
  punti di mercato (in un test leave-one-out sugli strike interni sbaglia in
  media 0,2 punti di vol, contro 0,7 della parabola);
- metodo di confronto "quadratic": parabola ai minimi quadrati pesati con la vega.
Fuori dall'intervallo degli strike quotati la vol resta costante al valore del
bordo (estrapolazione piatta). Con 2 punti si usa una retta, con 1 una costante.

Febbraio 2021: le opzioni Apr21 scadono il 23/2 e nelle ultime settimane quasi
tutte valgono pochi centesimi, quindi vengono scartate dai filtri. Se una gamba
ha meno di smile_min_points opzioni valide, si usa lo smile (in log-moneyness)
dell'altra gamba dello stesso giorno; se manca anche quello, lo smile del giorno
prima. Le due gambe hanno vol quasi identiche (ATM: differenza di pochi decimi).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.stats import norm

from dataclasses import dataclass

from config import CollarConfig, IVFilters
from data import MarketData


# ------------------------------------------------------------------ Black-76
def black76_price(F, K, T, sigma, is_call):
    """Prezzo di Black-76 con r = 0. Accetta numeri o array numpy."""
    F, K, T, sigma = map(np.asarray, (F, K, T, sigma))
    is_call = np.asarray(is_call, dtype=bool)
    sd = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sd**2) / sd
    d2 = d1 - sd
    call = F * norm.cdf(d1) - K * norm.cdf(d2)
    put = K * norm.cdf(-d2) - F * norm.cdf(-d1)
    return np.where(is_call, call, put)


def black76_vega(F, K, T, sigma):
    """dPrezzo/dσ (uguale per call e put)."""
    F, K, T, sigma = map(np.asarray, (F, K, T, sigma))
    sd = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sd**2) / sd
    return F * norm.pdf(d1) * np.sqrt(T)


def implied_vol(price, F, K, T, is_call, lo=1e-4, hi=5.0, n_iter=80):
    """Vol implicita per bisezione, su tutti gli elementi in parallelo.

    Il prezzo di Black-76 cresce con σ, quindi per ogni opzione esiste al più una
    σ che riproduce il prezzo. Si parte dall'intervallo [lo, hi] e lo si dimezza
    n_iter volte (80 dimezzamenti: precisione molto sotto 1e-10).

    Restituisce NaN quando il prezzo è fuori dai limiti di non arbitraggio:
      call: max(F-K, 0) < prezzo < F      put: max(K-F, 0) < prezzo < K
    """
    price, F, K, T = (np.asarray(x, dtype=float) for x in (price, F, K, T))
    is_call = np.asarray(is_call, dtype=bool)
    intrinsic = np.where(is_call, np.maximum(F - K, 0), np.maximum(K - F, 0))
    upper = np.where(is_call, F, K)
    valid = (price > intrinsic) & (price < upper) & (T > 0)

    a = np.full(price.shape, lo)
    b = np.full(price.shape, hi)
    Ts = np.where(T > 0, T, np.nan)   # evita divisioni per zero sulle opzioni scadute
    with np.errstate(invalid="ignore", divide="ignore"):
        for _ in range(n_iter):
            m = 0.5 * (a + b)
            too_high = black76_price(F, K, Ts, m, is_call) > price
            b = np.where(too_high, m, b)
            a = np.where(too_high, a, m)
    iv = 0.5 * (a + b)
    # se la soluzione è "incollata" a un estremo, il prezzo non è raggiungibile
    iv = np.where(valid & (iv > lo * 1.01) & (iv < hi * 0.99), iv, np.nan)
    return iv


# ------------------------------------------------------- vol di tutte le opzioni
def compute_implied_vols(md: MarketData, cfg: CollarConfig,
                         contracts: tuple[str, ...] | None = None) -> pd.DataFrame:
    """Vol implicite (al mid, al bid e all'ask) e filtri, per le opzioni sui contratti dati.

    Colonne aggiunte alla tabella delle opzioni:
      moneyness  = K / F
      intrinsic  = valore intrinseco
      otm        = True se put con K < F o call con K >= F
      rel_spread = (ask - bid) / mid
      iv_mid, iv_bid, iv_ask
      reason     = motivo dell'esclusione ("" se l'opzione è tenuta)
      keep       = True se l'opzione supera tutti i filtri
    """
    f: IVFilters = cfg.iv_filters
    contracts = contracts or cfg.legs
    o = md.options[md.options.contract.isin(contracts)].copy()

    is_call = (o.cp == "C").to_numpy()
    o["moneyness"] = o.strike / o.F
    o["intrinsic"] = np.where(is_call, np.maximum(o.F - o.strike, 0), np.maximum(o.strike - o.F, 0))
    o["otm"] = np.where(is_call, o.strike >= o.F, o.strike < o.F)
    o["rel_spread"] = o.spread / o.mid

    for col, px in (("iv_mid", "mid"), ("iv_bid", "bid"), ("iv_ask", "ask")):
        o[col] = implied_vol(o[px].to_numpy(), o.F.to_numpy(), o.strike.to_numpy(),
                             o.ttm.to_numpy(), is_call)
    if cfg.model.smile_price == "settle":
        # smile dai prezzi di settlement: sono gli stessi prezzi con cui si valutano ogni giorno
        # le opzioni di copertura (margini), quindi collar e coperture sono marcati in modo coerente.
        # Dove il settlement manca o non dà una vol si tiene quella al mid.
        iv_set = implied_vol(o["settle"].to_numpy(dtype=float), o.F.to_numpy(), o.strike.to_numpy(),
                             o.ttm.to_numpy(), is_call)
        o["iv_quote_mid"] = o["iv_mid"]
        o["iv_mid"] = np.where(np.isfinite(iv_set), iv_set, o["iv_mid"])

    # filtri, nell'ordine: il motivo registrato è il primo che fallisce
    reason = pd.Series("", index=o.index, dtype=object)

    def flag(mask, text):
        reason[(reason == "") & mask] = text

    flag(o.bid.isna() | o.ask.isna() | (o.bid <= 0), "non quotata")
    if f.otm_only:
        flag(~o.otm, "ITM")
    flag(o.ttm_days < f.min_ttm_days, "troppo vicina alla scadenza")
    flag(o.mid < f.min_price, "prezzo troppo basso")
    flag(o.rel_spread > f.max_rel_spread, "spread troppo largo")
    flag(o.iv_mid.isna(), "vol non calcolabile")
    flag((o.iv_mid < f.min_iv) | (o.iv_mid > f.max_iv), "vol fuori intervallo")

    o["reason"] = reason
    o["keep"] = reason == ""
    return o.reset_index(drop=True)


def filter_summary(iv: pd.DataFrame) -> pd.DataFrame:
    """Quante opzioni escono per ciascun filtro, per contratto."""
    t = iv.assign(reason=iv.reason.replace("", "TENUTA")).pivot_table(
        index="reason", columns="contract", values="strike", aggfunc="size", fill_value=0)
    return t.sort_values(t.columns[0], ascending=False)


# ------------------------------------------------------------------ vol ATM
def atm_vol(iv: pd.DataFrame) -> pd.DataFrame:
    """Vol implicita a K = F, interpolando linearmente tra i due strike tenuti più vicini.

    È solo un indicatore rapido del livello di volatilità giorno per giorno:
    la stima vera arriverà con lo smile.
    Restituisce una tabella data × contratto.
    """
    kept = iv[iv.keep]
    out = {}
    for (d, c), g in kept.groupby(["date", "contract"]):
        g = g.sort_values("strike")
        F = g.F.iloc[0]
        below, above = g[g.strike <= F], g[g.strike >= F]
        if below.empty or above.empty:
            continue
        k1, v1 = below.strike.iloc[-1], below.iv_mid.iloc[-1]
        k2, v2 = above.strike.iloc[0], above.iv_mid.iloc[0]
        out[(d, c)] = v1 if k1 == k2 else v1 + (v2 - v1) * (F - k1) / (k2 - k1)
    s = pd.Series(out, name="atm_vol")
    s.index.names = ["date", "contract"]
    return s.unstack("contract")


# ---------------------------------------------------------------------- smile
@dataclass
class Smile:
    """Smile di un giorno e di un futures, come funzione della log-moneyness x = ln(K/F)."""
    xs: np.ndarray            # nodi (log-moneyness delle opzioni usate), crescenti
    vols: np.ndarray          # vol implicite ai nodi (per "pchip") ...
    coef: np.ndarray | None   # ... oppure coefficienti a, b, c della parabola (per "quadratic")
    method: str
    source: str = "propria"   # "propria", "dall'altra gamba", "dal giorno prima"
    wing_decay: float = 0.0   # >0: estrapolazione C¹ con tanh smorzata (evita kinks di delta sul bordo)

    def __post_init__(self):
        self._f = None
        if self.method == "pchip" and len(self.xs) >= 3:
            self._f = PchipInterpolator(self.xs, self.vols, extrapolate=False)
        self._s_lo = 0.0
        self._s_hi = 0.0
        self._v_min = float(self._eval_clamped(self.x_min))
        self._v_max = float(self._eval_clamped(self.x_max))
        if self.wing_decay > 0 and (self.x_max - self.x_min) > 2e-4:
            eps = 1e-4
            s_lo = (float(self._eval_clamped(self.x_min + eps)) - self._v_min) / eps
            s_hi = (self._v_max - float(self._eval_clamped(self.x_max - eps))) / eps
            self._s_lo = float(np.clip(s_lo, -0.5, 0.5))
            self._s_hi = float(np.clip(s_hi, -0.5, 0.5))

    def _eval_clamped(self, x):
        xc = np.clip(np.asarray(x, dtype=float), self.x_min, self.x_max)
        if self.method == "quadratic":
            return self.coef[0] + self.coef[1] * xc + self.coef[2] * xc**2
        if self._f is not None:
            return self._f(xc)
        return np.interp(xc, self.xs, self.vols)

    @property
    def x_min(self) -> float:
        return float(self.xs[0])

    @property
    def x_max(self) -> float:
        return float(self.xs[-1])

    @property
    def n_points(self) -> int:
        return len(self.xs)

    def vol_x(self, x):
        """Vol alla log-moneyness x (numero o array); piatta o C¹-smorzata fuori da [x_min, x_max]."""
        x_arr = np.asarray(x, dtype=float)
        v = self._eval_clamped(x_arr)
        if self.wing_decay > 0:
            lo = x_arr < self.x_min
            hi = x_arr > self.x_max
            if np.any(lo):
                v = np.where(lo, self._v_min + self._s_lo * self.wing_decay
                             * np.tanh((x_arr - self.x_min) / self.wing_decay), v)
            if np.any(hi):
                v = np.where(hi, self._v_max + self._s_hi * self.wing_decay
                             * np.tanh((x_arr - self.x_max) / self.wing_decay), v)
        return np.maximum(v, 0.01)

    def vol(self, K, F):
        """Vol allo strike K quando il futures vale F."""
        return self.vol_x(np.log(np.asarray(K, dtype=float) / F))

    @property
    def atm(self) -> float:
        return float(self.vol_x(0.0))

    def relabel(self, source: str) -> "Smile":
        return Smile(self.xs, self.vols, self.coef, self.method, source, self.wing_decay)


def fit_smile(x, iv, weights, method: str = "pchip", wing_decay: float = 0.0) -> Smile:
    """Costruisce lo smile dai punti (x = ln(K/F), vol implicita, peso).

    "pchip": interpolazione monotona a tratti (PCHIP) che passa esattamente per i
             punti, senza oscillazioni tra un nodo e l'altro. Scelta principale.
    "quadratic": parabola ai minimi quadrati pesati (pesi = vega). Più liscia ma
             meno precisa: usata solo come confronto.
    Fuori dall'intervallo dei nodi la vol è piatta (wing_decay = 0) oppure
    smorzata in modo C¹ verso un asintoto vicino al bordo (wing_decay > 0).
    """
    x, iv, w = (np.asarray(v, dtype=float) for v in (x, iv, weights))
    order = np.argsort(x)
    x, iv, w = x[order], iv[order], w[order]
    # strike duplicati (stessa x): si tiene la media
    xu, inv = np.unique(x, return_inverse=True)
    ivu = np.bincount(inv, weights=iv) / np.bincount(inv)
    wu = np.bincount(inv, weights=w)
    if len(xu) == 0:
        raise ValueError("Nessun punto per lo smile")
    coef = None
    if method == "quadratic":
        deg = min(2, len(xu) - 1)
        sw = np.sqrt(np.maximum(wu, 1e-12))
        A = np.vander(xu, deg + 1, increasing=True) * sw[:, None]
        coef = np.linalg.lstsq(A, ivu * sw, rcond=None)[0]
        coef = np.concatenate([coef, np.zeros(3 - len(coef))])
    return Smile(xs=xu, vols=ivu, coef=coef, method=method, wing_decay=wing_decay)


def calibrate_smiles(iv: pd.DataFrame, cfg: CollarConfig) -> dict:
    """Uno smile per ogni (giorno, gamba). Restituisce {(data, contratto): Smile}.

    iv è il risultato di compute_implied_vols. Vedi la nota in cima al file per la
    regola usata quando una gamba ha troppe poche opzioni.
    """
    nmin = cfg.model.smile_min_points
    kept = iv[iv.keep].copy()
    kept["x"] = np.log(kept.strike / kept.F)
    kept["w"] = black76_vega(kept.F, kept.strike, kept.ttm, kept.iv_mid)
    groups = {k: g for k, g in kept.groupby(["date", "contract"])}

    smiles: dict = {}
    last: dict = {}
    for d in sorted(iv.date.unique()):
        d = pd.Timestamp(d)
        own = {}
        for c in cfg.legs:
            g = groups.get((d, c))
            if g is not None and len(g) >= nmin:
                own[c] = fit_smile(g.x, g.iv_mid, g.w, cfg.model.smile_method, cfg.model.wing_decay)
        for c in cfg.legs:
            other = cfg.legs[1] if c == cfg.legs[0] else cfg.legs[0]
            if c in own:
                s = own[c]
            elif other in own:
                s = own[other].relabel("dall'altra gamba")
            elif c in last:
                s = last[c].relabel("dal giorno prima")
            else:
                continue
            smiles[(d, c)] = s
            last[c] = s
    return smiles


def smile_table(smiles: dict) -> pd.DataFrame:
    """Riepilogo degli smile: una riga per (giorno, gamba) con parametri e vol ATM."""
    rows = [dict(date=d, contract=c, x_min=s.x_min, x_max=s.x_max, n_points=s.n_points,
                 source=s.source, atm=s.atm) for (d, c), s in smiles.items()]
    return pd.DataFrame(rows).sort_values(["date", "contract"]).reset_index(drop=True)
