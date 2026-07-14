# Credit Retention Optimizer — CLAUDE.md

## Propósito del proyecto
Proyecto de consultoría para optimizar el presupuesto de retención de clientes:
1. **Churn prediction** — clasificar clientes en riesgo de fuga
2. **CLV estimation** — estimar el valor de cada cliente
3. **Budget optimization** — asignar el presupuesto de retención de forma óptima

## Estructura del paquete `retention_optimizer`
| Módulo | Responsabilidad |
|---|---|
| `preprocessing/` | Limpieza y feature engineering. `preprocess()` es el punto de entrada; los pipelines individuales viven en `preprocessing/pipelines/` (`cleaning.py`, `feature_engineering.py`) y se combinan en `preprocessing/preprocessing.py` |
| `models/` | Modelos de churn (clasificación) y CLV (regresión) |
| `optimization/` | Algoritmo de asignación de presupuesto |
| `evaluation/` | Métricas y función de coste de negocio |

## Entorno
- Gestor: **uv** (`uv sync`, `uv sync --extra dev`)
- Python: 3.11+
- Intérprete: `.venv/bin/python`

## Comandos clave
```bash
uv sync --extra dev      # instalar todo (core + dev)
pytest                   # ejecutar tests
ruff check src/          # linting
black src/               # formateo
jupyter lab              # notebooks
streamlit run app/app.py # dashboard
```

## Convenciones
- Una notebook por bloque funcional (EDA, features, churn, CLV, optimización)
- Modelos serializados en `models/` (no versionados)
- Figuras de reportes en `reports/figures/`
- Decisiones de diseño documentadas en `docs/decisions/`
