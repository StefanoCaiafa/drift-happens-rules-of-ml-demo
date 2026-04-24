# Drift Happens – Rules of ML Demo

Este proyecto demuestra buenas prácticas de Machine Learning en producción, basado en el paper:

**Rules of Machine Learning – Martin Zinkevich (Google)**

El objetivo es mostrar, de forma simple y reproducible, algunos de los problemas más comunes en sistemas reales de ML, especialmente la diferencia entre resultados offline y comportamiento en producción.

🔗 Referencia oficial:  
https://developers.google.com/machine-learning/guides/rules-of-ml?hl=es-419  

🎥 Video recomendado:  
https://www.youtube.com/watch?v=VfcY0edoSLU  

---

## Objetivo de la demo

Esta demo utiliza un problema sintético de churn para ilustrar dos ideas centrales del paper:

1. **Empezar con modelos simples e interpretables**
2. **Un modelo que funciona bien offline puede degradarse en producción**

Se busca mostrar que el verdadero desafío en ML no es solo entrenar modelos, sino construir sistemas robustos que funcionen correctamente en el tiempo.

---

## Qué muestra el notebook

📓 Notebook: `drift_happens_demo.ipynb`

El notebook está dividido en dos demos principales:

---

### Demo 1 – Empezar simple (Rule #4 + Rule #14)

Se comparan dos modelos:

- Logistic Regression (baseline simple e interpretable)
- Random Forest (modelo más flexible)

**Resultado observado:**

- Ambos modelos tienen performance muy similar  
- El modelo simple incluso puede rendir igual o mejor  

 Esto refuerza que:

> No es necesario empezar con modelos complejos para obtener buenos resultados.

Además, los modelos simples son más fáciles de:

- entender  
- debuggear  
- mantener  
- explicar en producción  

---

### Demo 2 – Offline vs Producción (Rule #37)

Se evalúa el mismo modelo en dos escenarios:

- Datos de validación (offline)  
- Datos simulando producción con drift  

Para simular producción se introduce:

- cambio en distribución de features  
- escalamiento inconsistente (train-serving skew)  
- ruido gaussiano  
- leve cambio en etiquetas  

**Resultado observado:**

- Alta accuracy offline (~0.98)  
- Caída significativa en serving (~0.89)  

 Esto demuestra que:

> Buenas métricas offline no garantizan buen rendimiento en producción.

Además, se incluye una función para detectar skew comparando medias de features, generando alertas cuando hay cambios importantes.

---

## Conceptos clave que ilustra la demo

- **Baseline primero**: un modelo simple es un excelente punto de partida  
- **Interpretabilidad**: facilita debugging y mantenimiento  
- **Train-serving skew**: diferencias entre entrenamiento y producción  
- **Data drift**: cambios en los datos con el tiempo  
- **Monitoreo**: esencial para sistemas en producción  

---

## Datos de la demo

Los datos son completamente sintéticos (no se usan archivos externos):

- Se generan con `sklearn.datasets.make_classification`  
- Representan un problema típico de churn  
- Permiten controlar el comportamiento del sistema  

El drift en producción se simula modificando:

- medias de ciertas features  
- escalas  
- ruido  
- distribución de etiquetas  

Esto hace que la demo sea:

- reproducible  
- simple de entender  
- representativa de problemas reales  

---

## Setup (Poetry)

### 1. Instalar Poetry (una sola vez)

```bash
pip install poetry
```

### 2. Instalar dependencias

```bash
poetry install
```

### 3. Ejecutar el notebook

Abrir el archivo:

`drift_happens_demo.ipynb`

Ejecutar las celdas en orden (de arriba hacia abajo).

Opcional, para abrir Jupyter con el entorno de Poetry:

```bash
poetry run jupyter notebook
```

---

## Estructura del proyecto

```text
.
├── drift_happens_demo.ipynb
├── pyproject.toml
├── poetry.lock
├── poetry.toml
└── README.md
```

---

## Conclusión

Esta demo refleja una de las ideas más importantes del paper:

> El problema real del Machine Learning no es el modelo, sino el sistema.

Un modelo puede funcionar perfectamente en pruebas, pero si los datos cambian o el pipeline no está bien diseñado, el rendimiento en producción se degrada.

Por eso, en ML en producción es clave:

- construir pipelines confiables
- monitorear constantemente
- iterar sobre datos y features