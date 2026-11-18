import numpy as np
import pandas as pd

from src import lead_scoring as ls


def test_captura_dentro_de_cada_mes():
    y = np.array([1, 0, 0, 0, 1, 1, 0, 0])
    puntaje = np.array([0.9, 0.1, 0.2, 0.3, 0.1, 0.8, 0.9, 0.2])
    grupo = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    r = ls.captura(y, puntaje, grupo, 0.25)        # una llamada por mes
    assert r["llamadas"] == 2 and r["ventas"] == 1 and np.isclose(r["parte_de_las_ventas"], 1 / 3)
    r = ls.captura(y, puntaje, grupo, 0.5)         # dos llamadas por mes
    assert r["ventas"] == 2


def test_auc_por_mes_ignora_meses_sin_variacion():
    y = np.array([1, 0, 1, 1])
    p = np.array([0.9, 0.1, 0.5, 0.4])
    g = np.array([1, 1, 2, 2])                     # el mes 2 solo tiene compras: no cuenta
    assert ls.auc_por_mes(y, p, g) == 1.0


def test_division_temporal_no_se_cruza():
    d = ls.cargar()
    tr, ca, te = ls.dividir(d)
    assert tr["orden"].max() < ca["orden"].min() and ca["orden"].max() < te["orden"].min()
    assert len(tr) + len(ca) + len(te) == len(d) == 41188
    assert "duration" not in ls.NUMERICAS + ls.CATEGORICAS
