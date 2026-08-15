# Detección de Neumonía Bacteriana y Viral mediante CNN

Clasificación de radiografías de tórax en 3 categorías (normal,
neumonía bacteriana, neumonía viral) mediante transfer learning
con DenseNet121, usando el dataset público de Kaggle
[`chest-xray-pneumonia`](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).

---

## Estructura del proyecto

El pipeline está dividido en **3 notebooks independientes**, pensados para
correrse en orden. Cada uno se puede cerrar y volver a abrir sin depender
de que los otros sigan con el kernel activo — todo lo que un notebook
necesita del anterior queda guardado en disco, no solo en memoria.

```
project/
├── 01_Preprocesamiento.ipynb   # Descarga/organiza el dataset, split 80/10/10
├── 02_Entrenamiento.ipynb      # Pipeline tf.data, entrenamiento (3 fases), guarda el modelo
├── 03_Validacion.ipynb         # Métricas, matrices de confusión, ROC-AUC, Grad-CAM
├── chest_xray/                 # Dataset de Kaggle (sin procesar)
└── outputs/                    # Se genera automáticamente al correr los notebooks
    ├── data/                   # train_df.csv, val_df.csv, test_df.csv, classes.txt
    ├── *.keras                 # Modelos entrenados (fase 1, fine-tuning, focal loss)
    ├── *.png                   # Gráficos (distribución, curvas, matrices, ROC-AUC)
    ├── *.csv                   # Reportes de inferencia y métricas
    └── gradcam_outputs/        # Mapas de calor individuales
```

### Flujo de datos entre notebooks

```
01_Preprocesamiento.ipynb
        ↓ guarda train_df.csv, val_df.csv, test_df.csv, classes.txt
        ↓ en outputs/data/
02_Entrenamiento.ipynb
        ↓ carga esos CSVs, entrena, guarda los .keras
        ↓ en outputs/
03_Validacion.ipynb
        ↓ carga el .keras y test_df.csv
        ↓ genera métricas, matrices, ROC-AUC, Grad-CAM
```

---

## 1. Requisitos

- Python **3.10** (requerido específicamente para `tensorflow-directml-plugin`; no funciona con 3.11+)
- Windows (para aceleración por GPU vía DirectML; funciona con AMD, Intel o NVIDIA)
- ~2 GB libres para el dataset + entorno virtual

## 2. Configuración del entorno

```powershell
py -3.10 -m venv venv_neumonia
venv_neumonia\Scripts\activate
python -m pip install --upgrade pip
pip install --no-cache-dir tensorflow-cpu==2.10.0 tensorflow-directml-plugin "numpy<2" "protobuf==3.19.6" scikit-learn opencv-python-headless matplotlib seaborn pandas jupyter ipykernel
python -m ipykernel install --user --name=venv_neumonia --display-name "Python 3.10 (Neumonia GPU)"
```

**Verificar que detecta la GPU antes de abrir los notebooks:**

```powershell
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Si imprime tu GPU (`PhysicalDevice(name='/physical_device:GPU:0', ...)`),
el entorno está listo.

> **Estas versiones exactas están fijadas a propósito.** `tensorflow-cpu==2.10.0`
> es la última versión compatible con `tensorflow-directml-plugin`. Instalar
> `tensorflow` a secas (sin fijar versión) o actualizar `numpy`/`protobuf`
> sueltos rompe el plugin de GPU — si necesitas instalar algo más en este
> entorno, siempre especifica las versiones exactas de arriba.

## 3. Dataset

Descarga [`chest-xray-pneumonia`](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
de Kaggle y descomprime la carpeta `chest_xray/` (con subcarpetas
`train/`, `val/`, `test/`) junto a los notebooks.

El notebook `01_Preprocesamiento.ipynb` **re-etiqueta** las imágenes de
`PNEUMONIA` en `bacteria`/`virus` a partir del nombre de archivo (Kaggle
ya los distingue así, ej. `person1_bacteria_1.jpeg`), y hace su **propio**
split 80/10/10 estratificado (descarta el split original de Kaggle, que
está desbalanceado).

## 4. Cómo correr

1. **`01_Preprocesamiento.ipynb`** — genera la estructura de 3 clases, el split, y los `.csv` en `outputs/data/`.
2. **`02_Entrenamiento.ipynb`** — entrena en 3 fases:
   - **Fase 1**: solo la cabecera (base DenseNet121 100% congelada), 20 épocas máx.
   - **Fase 2 (fine-tuning)**: se descongelan las últimas 100 capas de la base (BatchNorm siempre congelada), LR 10× menor.
   - **Fase 3 (Focal Loss, opcional)**: ronda adicional con Focal Loss (γ=2.0) para reforzar las clases más difíciles de distinguir.
3. **`03_Validacion.ipynb`** — carga el mejor modelo disponible automáticamente (Focal Loss > fine-tuning > fase 1) y genera:
   - `classification_report` (precisión/recall/F1 por clase)
   - Matriz de confusión 3×3
   - Matrices de confusión One-vs-Rest (TP/FP/TN/FN por clase)
   - Curvas ROC-AUC One-vs-Rest
   - Mapas de calor Grad-CAM (predicciones correctas e incorrectas)

## 5. Arquitectura del modelo

```
Imagen 224×224×3
    → DenseNet121 (preentrenada en ImageNet, transfer learning)
    → Global Average Pooling 2D
    → Dense(128, ReLU) + BatchNormalization + Dropout(0.4)
    → Dense(3, Softmax)
    → [P(normal), P(bacteria), P(virus)]
```

- **Optimizador:** Adam (LR 1e-4 fase 1 → 1e-5 fine-tuning → 5e-6 focal loss)
- **Pérdida:** `categorical_crossentropy` (fases 1-2) / Focal Loss (fase 3, opcional)
- **Balanceo de clases:** `class_weight` balanceado (`sklearn.compute_class_weight`), con refuerzo manual ×1.3 en "virus" durante fine-tuning
- **Regularización:** Dropout 0.4, EarlyStopping, ReduceLROnPlateau

## 6. Resultados de referencia

| Clase | Precisión | Recall | F1 | AUC |
|---|---|---|---|---|
| normal | 0.92 | 0.98 | 0.95 | 0.995 |
| bacteria | 0.85 | 0.77 | 0.81 | 0.912 |
| virus | 0.65 | 0.71 | 0.68 | 0.873 |

**Exactitud general: ~81%.** El principal reto identificado (consistente
con la literatura citada en la propuesta original) es la distinción entre
neumonía bacteriana y viral — el ~97% del error del modelo es confusión
cruzada entre estas dos clases, con muy poca confusión respecto a "normal".


