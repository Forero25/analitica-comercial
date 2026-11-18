"""Lead scoring con el dataset público Bank Marketing (Moro, Cortez y Rita, 2014; UCI, CC BY 4.0).

41.188 llamadas de una campaña de depósitos a plazo de un banco portugués, ordenadas por fecha
(mayo de 2008 a noviembre de 2010). La pregunta no es «¿qué tan bueno es el modelo?» sino
«si el equipo solo alcanza a llamar a una parte de la lista, ¿cuántas ventas trae empezar por
donde dice el modelo?».

Diseño, fijado antes de mirar la prueba:
- Fuera `duration`: la duración de la llamada solo se conoce después de llamar (la propia
  documentación del dataset lo advierte).
- Fuera las cinco variables macroeconómicas y el mes: son iguales para todos los leads de un mismo
  mes, así que no sirven para decidir a quién llamar primero. Dicen cuántos van a comprar, no
  quiénes. (Con ellas, un modelo entrenado en 2008 ordena al revés en 2010.)
- La lista se arma dentro de cada mes de campaña (26 tramos consecutivos con el mismo índice de
  precios mensual): así trabaja un equipo, priorizando entre los leads que tiene hoy.
- División temporal por meses de campaña: los 9 primeros entrenan (mayo 2008 a abril 2009), el
  décimo calibra y los 16 últimos (junio 2009 a noviembre 2010) son la prueba.
- Referencias: llamar en el orden de llegada (prospección sin priorizar) y una regla de vendedor
  (primero quien ya compró en una campaña anterior, después quien ya fue contactado).
- Hiperparámetros fijos. Semilla fija.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAIZ = pathlib.Path(__file__).resolve().parent.parent
DATOS = RAIZ / "data" / "raw" / "bank-additional-full.csv"
SALIDA = RAIZ / "publicado"
SEMILLA = 2026
CAPACIDADES = [0.05, 0.10, 0.20, 0.30, 0.50]

CATEGORICAS = ["job", "marital", "education", "default", "housing", "loan", "contact", "day_of_week", "poutcome"]
NUMERICAS = ["age", "intentos_previos", "contactado_antes", "previous"]
ETIQUETAS = {
    "job": "ocupación", "marital": "estado civil", "education": "educación", "default": "mora", "housing": "crédito de vivienda",
    "loan": "crédito personal", "contact": "canal", "day_of_week": "día", "poutcome": "campaña anterior",
    "age": "edad", "intentos_previos": "llamadas previas en esta campaña", "contactado_antes": "contactado en otra campaña",
    "previous": "contactos en campañas anteriores", "emp.var.rate": "variación del empleo", "cons.price.idx": "precios al consumidor",
    "cons.conf.idx": "confianza del consumidor", "euribor3m": "tasa Euribor", "nr.employed": "empleados en la economía",
}


FRASES = {
    ("contactado_antes", True): "ya fue contactado en otra campaña", ("contactado_antes", False): "nunca contactado antes",
    ("intentos_previos", False): "pocas llamadas en esta campaña", ("intentos_previos", True): "muchas llamadas en esta campaña",
    ("age", False): "menor edad", ("age", True): "mayor edad",
    ("previous", True): "varios contactos en campañas anteriores", ("previous", False): "pocos contactos en campañas anteriores",
}
VALORES = {"student": "estudiante", "retired": "jubilado", "success": "compró", "failure": "no compró", "cellular": "celular",
           "telephone": "teléfono fijo", "single": "soltero", "admin.": "administrativo", "unknown": "sin dato"}


def cargar(ruta=DATOS) -> pd.DataFrame:
    d = pd.read_csv(ruta, sep=";")
    d["y"] = (d["y"] == "yes").astype(int)
    d["intentos_previos"] = d["campaign"] - 1          # `campaign` incluye la llamada actual
    d["contactado_antes"] = (d["pdays"] != 999).astype(int)
    d["orden"] = np.arange(len(d))                       # el archivo viene ordenado por fecha
    cambio = (d[["month", "cons.price.idx"]] != d[["month", "cons.price.idx"]].shift()).any(axis=1)
    d["mes_campana"] = cambio.cumsum()                   # 1..26, tramos consecutivos de un mismo mes
    return d


def dividir(d: pd.DataFrame):
    return d[d.mes_campana <= 9], d[d.mes_campana == 10], d[d.mes_campana >= 11]


def preprocesador() -> ColumnTransformer:
    return ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=30), CATEGORICAS),
                              ("num", StandardScaler(), NUMERICAS)])


def logistica():
    return make_pipeline(preprocesador(), LogisticRegression(C=0.5, max_iter=3000))


def boosting():
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=30, sparse_output=False), CATEGORICAS),
                             ("num", "passthrough", NUMERICAS)])
    return make_pipeline(pre, HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=300,
                                                             l2_regularization=1.0, min_samples_leaf=50, random_state=SEMILLA))


def captura(y: np.ndarray, puntaje: np.ndarray, grupo: np.ndarray, capacidad: float) -> dict:
    """En cada mes de campaña se llama solo al `capacidad` de sus leads, empezando por el puntaje más alto."""
    llamadas = ventas = 0
    for g in np.unique(grupo):
        m = grupo == g
        n = max(1, int(round(capacidad * m.sum())))
        top = np.argsort(-puntaje[m], kind="stable")[:n]
        llamadas += n
        ventas += int(y[m][top].sum())
    return {"capacidad": capacidad, "llamadas": llamadas, "ventas": ventas, "parte_de_las_ventas": ventas / max(1, y.sum()),
            "tasa": ventas / llamadas, "llamadas_por_venta": llamadas / max(1, ventas)}


def auc_por_mes(y, puntaje, grupo) -> float:
    """AUC dentro de cada mes, promediado con peso según sus ventas: qué tan bien ordena la lista de un mismo mes."""
    aucs, pesos = [], []
    for g in np.unique(grupo):
        m = grupo == g
        if 0 < y[m].sum() < m.sum():
            aucs.append(roc_auc_score(y[m], puntaje[m]))
            pesos.append(y[m].sum())
    return float(np.average(aucs, weights=pesos))


def ic_captura(y, puntaje, grupo, capacidad, reps=1000) -> list[float]:
    """Intervalo del 95 % de la parte de las ventas capturadas, remuestreando llamadas dentro de cada mes."""
    rng = np.random.default_rng(SEMILLA)
    idx_por_mes = [np.flatnonzero(grupo == g) for g in np.unique(grupo)]
    sims = []
    for _ in range(reps):
        i = np.concatenate([rng.choice(ix, len(ix)) for ix in idx_por_mes])
        sims.append(captura(y[i], puntaje[i], grupo[i], capacidad)["parte_de_las_ventas"])
    return [float(np.quantile(sims, 0.025)), float(np.quantile(sims, 0.975))]


def razones(modelo, fila: pd.DataFrame, k: int = 3) -> list[str]:
    """Las k variables que más empujan el puntaje de este lead hacia arriba (contribución al logit)."""
    pre, reg = modelo[0], modelo[-1]
    x = pre.transform(fila)
    x = x.toarray() if hasattr(x, "toarray") else x
    contrib = x[0] * reg.coef_[0]
    nombres = pre.get_feature_names_out()
    orden = np.argsort(-contrib)[:k]
    salida = []
    for j in orden:
        if contrib[j] <= 0:
            break
        bloque, var = nombres[j].split("__", 1)
        if bloque == "cat":
            base, _, valor = var.rpartition("_")
            salida.append(f"{ETIQUETAS.get(base, base)}: {VALORES.get(valor, valor)}")
        else:
            salida.append(FRASES.get((var, x[0, j] > 0), f"{ETIQUETAS.get(var, var)} {'alto' if x[0, j] > 0 else 'bajo'}"))
    return salida


def main() -> None:
    d = cargar()
    tr, ca, te = dividir(d)
    X = NUMERICAS + CATEGORICAS
    yte, gte = te["y"].to_numpy(), te["mes_campana"].to_numpy()

    log = logistica().fit(tr[X], tr["y"])
    gbm = boosting().fit(tr[X], tr["y"])
    rng = np.random.default_rng(SEMILLA)
    puntajes = {
        "orden_de_llegada": -te["orden"].to_numpy().astype(float),
        "regla_de_vendedor": (te["poutcome"] == "success").to_numpy() * 2.0 + te["contactado_antes"].to_numpy() + rng.random(len(te)) * 0.1,
        "logistica": log.predict_proba(te[X])[:, 1],
        "boosting": gbm.predict_proba(te[X])[:, 1],
    }
    nombres = {"orden_de_llegada": "Orden de llegada", "regla_de_vendedor": "Regla: primero quien ya compró",
               "logistica": "Regresión logística", "boosting": "Gradient boosting"}

    modelos = {}
    for k, p in puntajes.items():
        capt = []
        for c in CAPACIDADES:
            r = {a: (round(b, 4) if isinstance(b, float) else b) for a, b in captura(yte, p, gte, c).items()}
            if c == 0.2:
                r["ic95"] = [round(v, 4) for v in ic_captura(yte, p, gte, c)]
            capt.append(r)
        modelos[k] = {"nombre": nombres[k], "auc_por_mes": round(auc_por_mes(yte, p, gte), 4), "captura": capt}

    # La evaluación al azar promete más: con las variables macro y una división aleatoria, el mismo
    # tipo de modelo parece excelente porque «aprende» en qué meses se vendió más.
    macro = ["emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]
    pre_m = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=30), CATEGORICAS + ["month"]),
                               ("num", StandardScaler(), NUMERICAS + macro)])
    a_tr, a_te = train_test_split(d, test_size=0.2, random_state=SEMILLA, stratify=d["y"])
    cols = X + macro + ["month"]
    m_azar = make_pipeline(pre_m, LogisticRegression(C=0.5, max_iter=3000)).fit(a_tr[cols], a_tr["y"])
    auc_azar = roc_auc_score(a_te["y"], m_azar.predict_proba(a_te[cols])[:, 1])
    m_tiempo = make_pipeline(pre_m, LogisticRegression(C=0.5, max_iter=3000)).fit(tr[cols], tr["y"])
    auc_tiempo_macro = roc_auc_score(yte, m_tiempo.predict_proba(te[cols])[:, 1])

    # Cuánto compra la gente cambia mucho de un periodo a otro: la probabilidad absoluta del modelo
    # no sirve tal cual; lo que se sostiene es el orden.
    niveles = {"entrenamiento": round(float(tr["y"].mean()), 4), "calibracion": round(float(ca["y"].mean()), 4),
               "prueba": round(float(yte.mean()), 4)}

    # La lista del día: los 12 leads con más puntaje del primer mes de la prueba, con sus razones.
    mes = te[te.mes_campana == te.mes_campana.min()]
    p_mes = log.predict_proba(mes[X])[:, 1]
    top = np.argsort(-p_mes)[:12]
    lista = [{"prioridad": i + 1, "lead": f"L-{int(mes.iloc[j]['orden']):05d}",
              "veces_el_promedio": round(float(p_mes[j] / p_mes.mean()), 1),
              "por_que": razones(log, mes.iloc[[j]][X]), "compro": bool(mes.iloc[j]["y"])} for i, j in enumerate(top)]
    lista_info = {"leads_del_mes": int(len(mes)), "ventas_del_mes": int(mes["y"].sum()),
                  "ventas_en_la_lista": int(sum(l["compro"] for l in lista))}

    # Coeficientes de la logística: qué sube y qué baja la prioridad.
    pre, reg = log[0], log[-1]
    coef = sorted(zip(pre.get_feature_names_out(), reg.coef_[0]), key=lambda t: -abs(t[1]))[:10]
    coeficientes = [{"variable": n.split("__", 1)[1], "efecto": round(float(np.exp(b)), 3)} for n, b in coef]

    SALIDA.mkdir(exist_ok=True)
    resumen = {
        "fuente": "Bank Marketing (Moro, Cortez y Rita, 2014), UCI Machine Learning Repository, CC BY 4.0",
        "llamadas": int(len(d)), "ventas": int(d["y"].sum()), "meses_de_campana": int(d["mes_campana"].max()),
        "periodos": {"entrenamiento": int(len(tr)), "calibracion": int(len(ca)), "prueba": int(len(te)),
                     "meses_prueba": int(te["mes_campana"].nunique())},
        "tasa_de_compra": niveles,
        "modelos": modelos,
        "auc_con_macro": {"division_al_azar": round(float(auc_azar), 4), "division_temporal": round(float(auc_tiempo_macro), 4)},
        "coeficientes_logistica": coeficientes,
        "lista_del_dia": lista, "lista_info": lista_info,
    }
    (SALIDA / "lead_scoring.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("tasa de compra:", niveles, "| prueba:", resumen["periodos"])
    for k, m in modelos.items():
        c = {x["capacidad"]: x for x in m["captura"]}
        print(f"{m['nombre']:32s} AUC/mes {m['auc_por_mes']:.3f} | 10 %: {c[0.1]['parte_de_las_ventas']:.1%} | 20 %: {c[0.2]['parte_de_las_ventas']:.1%} "
              f"{c[0.2].get('ic95')} {c[0.2]['llamadas_por_venta']:.1f} llam/venta | 50 %: {c[0.5]['parte_de_las_ventas']:.1%}")
    print("AUC con macro: al azar", round(auc_azar, 3), "temporal", round(auc_tiempo_macro, 3))
    print(lista_info); [print(l) for l in lista[:6]]; print(coeficientes)


if __name__ == "__main__":
    main()
