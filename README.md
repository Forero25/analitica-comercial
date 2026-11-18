# Analítica comercial

Código y resultados de dos ensayos del portafolio de José Nicolás Forero Urrego. Todo con datos públicos.

## 1. Lead scoring que sí usan los vendedores

`src/lead_scoring.py` · resultados en `publicado/lead_scoring.json`

**Pregunta.** Si un equipo solo alcanza a llamar a una parte de sus leads, ¿cuántas ventas más trae empezar por donde dice un modelo?

**Datos.** Bank Marketing (Moro, Cortez y Rita, 2014; UCI Machine Learning Repository, CC BY 4.0): 41.188 llamadas de una campaña de depósitos a plazo, ordenadas por fecha de mayo de 2008 a noviembre de 2010.

**Diseño, fijado antes de mirar la prueba**
- Sin `duration` (se conoce después de la llamada) y sin las variables macroeconómicas ni el mes: son iguales para todos los leads de un mismo mes, así que dicen cuántos compran, no quiénes.
- La lista se arma dentro de cada uno de los 26 meses de campaña, como trabaja un equipo con los leads que tiene hoy.
- División temporal: 9 meses entrenan, 1 calibra y los 16 más recientes son la prueba.
- Referencias: el orden de llegada y una regla de vendedor (primero quien ya compró en una campaña anterior).
- Métrica principal: qué parte de las ventas del mes se captura llamando solo al 20 % de la lista, con intervalo del 95 % por remuestreo.

## 2. Un dashboard no es una decisión

`src/tablero_decision.py` · resultados en `publicado/tablero_decision.json`

**Pregunta.** Con el mismo dato (el porcentaje de contratación directa de cada alcaldía), ¿qué cambia entre mostrarlo en un tablero y usarlo para decidir qué 20 alcaldías revisar?

**Datos.** Los agregados del [observatorio de contratación pública](https://github.com/Forero25/observatorio-secop-datos) (SECOP II, CC BY-SA 4.0), que el script descarga.

**Método.** El tablero ordena por porcentaje. La decisión encoge cada porcentaje hacia el de sus pares según cuántos contratos tiene (Bayes empírico, beta-binomial) y prioriza por contratos directos por encima de la mediana. Se compara qué tanto se sostiene cada lista de un año al siguiente.

Son señales estadísticas para orientar una revisión, no acusaciones; el análisis no nombra municipios.

## Correrlo

```bash
uv sync
uv run pytest -q
uv run python -m src.lead_scoring
uv run python -m src.tablero_decision
```

## Lo que no hace

- El lead scoring usa una sola campaña de un solo banco y un producto; no sabe nada del vendedor, del guion ni del costo de cada llamada.
- La lista de la decisión favorece a las alcaldías grandes por construcción: prioriza por volumen en juego, que es una elección, no un hallazgo.
- Ninguno de los dos mide el efecto de usar la herramienta; eso requiere un experimento con el equipo.
