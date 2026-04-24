# Drift Happens - Rules of ML Demo

Buenas prácticas de ML en producción, basado en:

- Rules of Machine Learning (Martin Zinkevich)

Este proyecto presenta un notebook reproducible con datos sintéticos de churn para ilustrar dos ideas clave: empezar con modelos simples e interpretables, y monitorear el train-serving skew cuando el modelo pasa a producción.

Referencia oficial:
<a href="https://developers.google.com/machine-learning/guides/rules-of-ml?hl=es-419" target="_blank" rel="noopener noreferrer">Reglas del aprendizaje automático: prácticas recomendadas para la ingeniería de AA</a>

Video recomendado (Martin Zinkevich):
<a href="https://www.youtube.com/watch?v=VfcY0edoSLU" target="_blank" rel="noopener noreferrer">Rules of Machine Learning - charla introductoria</a>

La demo cubre 2 bloques:

1. Rule #4 + Rule #14:
   - Keep the first model simple.
   - Start with an interpretable model.
2. Rule #37:
   - Medir train-serving skew.

## Qué muestra el notebook

Notebook: `drift_happens_demo.ipynb`

- Demo 1 (offline): compara Regresión Logística vs Random Forest para mostrar que un baseline simple e interpretable puede ser suficiente.
- Demo 2 (serving): aplica drift sintético y evidencia caída de accuracy + alerta de skew por cambio de distribución de features.

Mensaje principal: en ML de producción, no alcanza con un buen resultado offline; hay que monitorear datos y comportamiento del sistema.

## Setup (Poetry)

1. Instalar Poetry (una sola vez):

```powershell
pip install poetry
```

2. Instalar dependencias:

```powershell
poetry install
```

3. Abrir y ejecutar el notebook:

- Archivo: `drift_happens_demo.ipynb`
- Ejecutar celdas en orden, de arriba hacia abajo.

Opcional, para abrir Jupyter con el entorno de Poetry:

```powershell
poetry run jupyter notebook
```

## Datos de la demo

Los datos son sintéticos (sin CSV externo ni API):

- `sklearn.datasets.make_classification` genera el dataset base de churn.
- Luego se simula serving drift con:
   - corrimiento de media en features,
   - cambio de escala,
   - ruido gaussiano,
   - leve perturbación de etiquetas.

Esto hace la demo reproducible, controlada y fácil de explicar.

## Mensajes clave

- Start simple.
- Good offline metrics are not enough.
- Monitor production drift/skew.

## Estructura del proyecto

```text
.
├── drift_happens_demo.ipynb
├── poetry.lock
├── poetry.toml
├── pyproject.toml
└── README.md
```
