# Drift Happens - Rules of ML Demo

Demo corta y reproducible para mostrar 2 ideas:

1. Un modelo simple puede rendir parecido a uno mas complejo.
2. El drift de datos puede degradar performance en produccion.

## Setup (unico flujo con Poetry)

1. Instalar Poetry (una sola vez):

```powershell
pip install poetry
```

2. Instalar dependencias del proyecto:

```powershell
poetry install
```

Con esta configuracion, Poetry crea el entorno virtual local en `.venv/`.

3. Abrir y ejecutar el notebook:

- Archivo: `drift_happens_demo.ipynb`
- Ejecutar celdas en orden, de arriba hacia abajo.

Si quieres correr el script por terminal:

```powershell
poetry run python main.py
```

## De donde salen los datos

Los datos **no** vienen de un CSV externo ni API.

Se generan de forma sintetica con:

- `sklearn.datasets.make_classification` para crear un dataset binario base.
- Una transformacion sobre ese dataset para simular drift:
  - corrimiento de media en algunas features,
  - cambio de escala en otra feature,
  - ruido gaussiano controlado.

Esto permite una demo deterministica y facil de explicar.

## Estructura

```text
.
├── drift_happens_demo.ipynb
├── main.py
├── poetry.lock
├── poetry.toml
├── pyproject.toml
└── README.md
```
