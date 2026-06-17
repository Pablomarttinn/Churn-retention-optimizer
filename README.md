# Credit Retention Optimizer

> Optimización de presupuesto de retención de clientes mediante predicción de churn, estimación de CLV y asignación óptima de recursos.

## Stack
- **Churn model**: XGBoost / scikit-learn
- **CLV model**: regresión (BG/NBD o similar)
- **Optimización**: programación lineal / heurística
- **Dashboard**: Streamlit

## Instalación
```bash
uv sync              # dependencias core
uv sync --extra dev  # + pytest, black, ruff
```

## Uso rápido
```bash
jupyter lab          # exploración en notebooks/
streamlit run app/app.py  # dashboard interactivo
```

## Estructura
```
src/retention_optimizer/
├── data/         # carga y limpieza
├── features/     # feature engineering
├── models/       # churn + CLV
├── optimization/ # asignación de presupuesto
└── evaluation/   # métricas y función de coste
```
