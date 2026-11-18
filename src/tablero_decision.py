"""El mismo dato como tablero y como decisión: contratación directa en las alcaldías.

Datos: los agregados que publica el observatorio de contratación pública
(github.com/Forero25/observatorio-secop-datos, CC BY-SA 4.0, a partir de SECOP II).

- El tablero ordena las alcaldías por su porcentaje de contratación directa. Arriba quedan
  municipios con muy pocos contratos: un 100 % con 4 contratos dice poco.
- La decisión es otra: «solo podemos revisar 20 alcaldías este año, ¿cuáles?». Para eso:
  1. Se encoge cada porcentaje hacia el de sus pares según cuántos contratos tiene (Bayes
     empírico con una beta ajustada a las alcaldías del año).
  2. Se prioriza por contratos directos por encima de lo típico: contratos × (porcentaje
     encogido − mediana). Eso combina qué tan atípico es con cuánto está en juego.
- Validación: ¿cuál de las dos listas se sostiene al año siguiente? Una lista que cambia por
  completo cada año no sirve para planear revisiones.

Son señales estadísticas para orientar una revisión, no acusaciones. El ensayo no nombra
municipios.
"""
from __future__ import annotations

import json
import pathlib
import urllib.request

import numpy as np
import pandas as pd
from scipy import optimize, special

RAIZ = pathlib.Path(__file__).resolve().parent.parent
CACHE = RAIZ / "data" / "observatorio"
SALIDA = RAIZ / "publicado"
ORIGEN = "https://github.com/Forero25/observatorio-secop-datos/releases/download/datos/"
ARCHIVOS = ["hechos_contratos.csv", "entidad.csv", "modalidad.csv", "territorio.csv"]
REVISIONES = 20
MIN_CONTRATOS_AJUSTE = 30


def bajar() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    for f in ARCHIVOS:
        if not (CACHE / f).exists():
            with urllib.request.urlopen(ORIGEN + f, timeout=180) as r:
                (CACHE / f).write_bytes(r.read())


def alcaldias_por_anio() -> pd.DataFrame:
    h = pd.read_csv(CACHE / "hechos_contratos.csv")
    h = h.merge(pd.read_csv(CACHE / "entidad.csv"), on="id_entidad").merge(pd.read_csv(CACHE / "modalidad.csv"), on="id_modalidad")
    h = h[h["clase_entidad"] == "Alcaldía"].copy()
    h["anio"] = h["fecha"].str[:4].astype(int)
    h["directos"] = np.where(h["seleccion"] == "Contratación directa", h["contratos"], 0)
    g = h.groupby(["anio", "id_territorio"], as_index=False)[["contratos", "directos"]].sum()
    g["pct"] = g["directos"] / g["contratos"]
    return g


def ajustar_beta(directos: np.ndarray, contratos: np.ndarray) -> tuple[float, float]:
    """Máxima verosimilitud de una beta-binomial: la distribución del porcentaje entre alcaldías."""
    def nll(t):
        a, b = np.exp(t)
        return -np.sum(special.betaln(directos + a, contratos - directos + b) - special.betaln(a, b))
    r = optimize.minimize(nll, x0=np.log([4.0, 1.0]), method="Nelder-Mead")
    return tuple(np.exp(r.x))


def priorizar(anio: pd.DataFrame) -> pd.DataFrame:
    base = anio[anio["contratos"] >= MIN_CONTRATOS_AJUSTE]
    a, b = ajustar_beta(base["directos"].to_numpy(), base["contratos"].to_numpy())
    mediana = float(base["pct"].median())
    t = anio.copy()
    t["pct_encogido"] = (t["directos"] + a) / (t["contratos"] + a + b)
    t["exceso"] = t["contratos"] * (t["pct_encogido"] - mediana)
    t["rank_tablero"] = t["pct"].rank(ascending=False, method="first")
    t["rank_decision"] = t["exceso"].rank(ascending=False, method="first")
    t.attrs.update({"a": a, "b": b, "mediana": mediana})
    return t


def main() -> None:
    bajar()
    g = alcaldias_por_anio()
    fechas = pd.read_csv(CACHE / "hechos_contratos.csv", usecols=["fecha"])["fecha"]
    ultimo_mes = pd.to_datetime(fechas).max()
    anio = ultimo_mes.year if ultimo_mes.month == 12 else ultimo_mes.year - 1   # último año completo
    t = priorizar(g[g["anio"] == anio])
    prev = priorizar(g[g["anio"] == anio - 1])

    tab = t.nsmallest(REVISIONES, "rank_tablero")
    dec = t.nsmallest(REVISIONES, "rank_decision")
    total_exceso = t.loc[t["exceso"] > 0, "exceso"].sum()

    # ¿Se sostiene la lista al año siguiente? De los 20 del año anterior que siguen con datos,
    # cuántos vuelven a quedar entre los 20 de este año.
    def persistencia(col):
        top_prev = set(prev.nsmallest(REVISIONES, col)["id_territorio"])
        top_act = set(t.nsmallest(REVISIONES, col)["id_territorio"])
        return len(top_prev & top_act)

    ambos = t.merge(prev, on="id_territorio", suffixes=("", "_prev"))
    resumen = {
        "fuente": "Observatorio de contratación pública (SECOP II, datos.gov.co), CC BY-SA 4.0",
        "anio": anio, "alcaldias": int(len(t)), "contratos": int(t["contratos"].sum()),
        "mediana_pct_directa": round(t.attrs["mediana"], 4),
        "beta": {"a": round(t.attrs["a"], 3), "b": round(t.attrs["b"], 3), "contratos_equivalentes": round(t.attrs["a"] + t.attrs["b"], 1)},
        "tablero": {
            "min_contratos_top": int(tab["contratos"].min()), "mediana_contratos_top": float(tab["contratos"].median()),
            "con_menos_de_30_contratos": int((tab["contratos"] < MIN_CONTRATOS_AJUSTE).sum()),
            "con_100_pct": int((tab["pct"] == 1).sum()),
            "contratos_cubiertos": int(tab["contratos"].sum()),
            "exceso_cubierto": round(float(tab["exceso"].clip(lower=0).sum() / total_exceso), 4),
        },
        "decision": {
            "min_contratos_top": int(dec["contratos"].min()), "mediana_contratos_top": float(dec["contratos"].median()),
            "con_menos_de_30_contratos": int((dec["contratos"] < MIN_CONTRATOS_AJUSTE).sum()),
            "pct_mediano_top": round(float(dec["pct"].median()), 4),
            "con_100_pct": int((dec["pct"] == 1).sum()),
            "contratos_cubiertos": int(dec["contratos"].sum()),
            "exceso_cubierto": round(float(dec["exceso"].clip(lower=0).sum() / total_exceso), 4),
        },
        "en_ambas_listas": int(len(set(tab["id_territorio"]) & set(dec["id_territorio"]))),
        "persistencia_un_anio": {"tablero": persistencia("rank_tablero"), "decision": persistencia("rank_decision"), "de": REVISIONES},
        "correlacion_entre_anios": {
            "pct_crudo": round(float(ambos[["pct", "pct_prev"]].corr(method="spearman").iloc[0, 1]), 3),
            "pct_encogido": round(float(ambos[["pct_encogido", "pct_encogido_prev"]].corr(method="spearman").iloc[0, 1]), 3),
            "alcaldias_en_ambos_anios": int(len(ambos)),
        },
    }
    SALIDA.mkdir(exist_ok=True)
    (SALIDA / "tablero_decision.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(resumen, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
