import numpy as np
import pandas as pd

from src import tablero_decision as td


def test_la_beta_recupera_parametros_conocidos():
    rng = np.random.default_rng(1)
    p = rng.beta(8, 2, 400)
    n = rng.integers(50, 500, 400)
    d = rng.binomial(n, p)
    a, b = td.ajustar_beta(d, n)
    assert abs(a / (a + b) - 0.8) < 0.02 and 5 < a + b < 16


def test_un_municipio_pequeno_se_encoge_y_no_encabeza_la_decision():
    filas = [{"id_territorio": i, "contratos": 200, "directos": 160} for i in range(40)]
    filas += [{"id_territorio": 100, "contratos": 3, "directos": 3},          # 100 % con 3 contratos
              {"id_territorio": 101, "contratos": 3000, "directos": 2850}]    # 95 % con 3.000 contratos
    t = pd.DataFrame(filas)
    t["pct"] = t["directos"] / t["contratos"]
    r = td.priorizar(t).set_index("id_territorio")
    assert r.loc[100, "rank_tablero"] == 1                      # el tablero lo pone primero
    assert r.loc[100, "pct_encogido"] < 0.95                    # pero con 3 contratos se encoge
    assert r.loc[101, "rank_decision"] == 1                     # la decisión prioriza lo que está en juego
