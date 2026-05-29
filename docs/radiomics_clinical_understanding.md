# Comprensión Clínica de Radiómicas en TC de Stroke

> **Documento de Investigación Médica** — Fase Inv  
> Fecha: 2026-05-29  
> Objetivo: Entender el significado médico de cada familia de features radiómicas y recrear el razonamiento del neurólogo

---

## 1. Fundamentos del Stroke en TC

### 1.1 Qué ve el neurólogo en una TC de stroke

Cuando un neurólogo evalúa una TC cerebral simple (NCCT) en un paciente con stroke isquémico agudo (< 4.5h), busca **signos de isquemia temprana**:

| Signo | Descripción | Significado clínico |
|-------|-------------|---------------------|
| **Pérdida de cinta insular** | Desaparición de la diferenciación gris/blanco en la corteza insular | Primera área en afectarse por isquemia (pobre colateral) |
| **Desaparición núcleo lentiforme** | Pérdida de definición del putamen/pálido | Isquemia en territorio MCA profundo |
| **Efecto de masa/edema** | Compresión de surcos/ventrículos | Edema citotóxico (daño irreversible incipiente) |
| **Hipodensidad** | Áreas más oscuras = menor atenuación | Acumulación de agua (Net Water Uptake) |
| **Arteria hiperdensa** | Arteria MCA más brillante = trombo | Oclusión de gran vaso |

### 1.2 ASPECTS — El "lenguaje" del neurólogo

El **ASPECTS** (Alberta Stroke Program Early CT Score) es un sistema de 10 puntos que cuantifica cambios isquémicos tempranos:

```
ASPECTS = 10 - (puntos perdidos por región afectada)

Regiones evaluadas (10):
├── Ganglionares (4 puntos)
│   ├── C (Cabeza del caudado)
│   ├── L (Núcleo lentiforme)
│   ├── IC (Cápsula interna)
│   └── I (Cinta insular)
└── Corticales (6 puntos) — territorio MCA
    ├── M1 (Corteza anterior MCA)
    ├── M2 (Corteza lateral a cinta insular)
    ├── M3 (Corteza posterior MCA)
    ├── M4 (Superior anterior)
    ├── M5 (Superior lateral)
    └── M6 (Superior posterior)
```

**Umbral crítico**: ASPECTS ≤ 7 → alto riesgo de pobre resultado/muerte  
**ASPECTS = 10** → TC normal (sin cambios isquémicos tempranos visibles)

---

## 2. Mapeo de Features Radiómicas → Hallazgos Visuales

### 2.1 Familia Shape2D (Forma 2D)

Estas features describen la **morfología de la lesión isquémica**:

| Feature | Descripción matemática | Interpretación clínica |
|---------|------------------------|------------------------|
| `shape2d_meshsurface` | Área de superficie de la lesión | **Volumen de tejido dañado**. Mayor área = lesión más extensa = más tiempo desde inicio |
| `shape2d_perimeter` | Perímetro de la lesión | **Bordes de la lesión**. Perímetro alto con área pequeña = lesión irregular/fragmentada |
| `shape2d_sphericity` | Qué tan esférica es la lesión | **Patrón de isquemia**. Esfera = núcleo infarto (irreversible); Irregular = penumbra (salvable) |
| `shape2d_elongation` | Relación eje mayor/menor | **Dirección del flujo**. Elongación alta = sigue distribución vascular MCA |
| `shape2d_maximumdiameter` | Diámetro máximo | **Extensión máxima**. Correlaciona con volumen de infarto final |

**Significado fisiopatológico**:
- Las lesiones **subagudas** (> 6h) tienden a ser más esféricas (edema citotóxico completo)
- Las lesiones **agudas** (< 3h) son más irregulares (penumbra predominante)

### 2.2 Familia FirstOrder (Intensidad/Histograma)

Estas features describen la **distribución de intensidades de Hounsfield** dentro de la lesión:

| Feature | Significado | Interpretación en stroke |
|---------|-------------|---------------------------|
| `firstorder_mean` | Media de HU en la lesión | **Grado de hipodensidad**. Menor valor = más edema/agua = más tiempo |
| `firstorder_median` | Mediana de HU | Similar a mean, más robusto a outliers |
| `firstorder_minimum` | Valor mínimo de HU | **Núcleo del infarto**. Valores muy bajos (< 15 HU) = daño irreversible |
| `firstorder_10percentile` | Percentil 10 | Cola inferior de distribución — núcleo isquémico |
| `firstorder_90percentile` | Percentil 90 | Cola superior — áreas de penumbra |
| `firstorder_entropy` | Entropía del histograma | **Heterogeneidad del tejido**. Alta entropía = mezcla de núcleo + penumbra |
| `firstorder_skewness` | Asimetría del histograma | Skewness negativo = cola hacia valores bajos (más edema) |
| `firstorder_kurtosis` | "Puntiagudez" del histograma | Kurtosis alta = distribución concentrada (lesión homogénea = tardía) |
| `firstorder_interquartilerange` | Rango intercuartílico | Variabilidad de intensidades dentro de la lesión |
| `firstorder_uniformity` | Uniformidad | Baja uniformidad = heterogéneo = lesión aguda con penumbra |

**Concepto clave: Net Water Uptake (NWU)**
```
NWU = (1 - HU_lesión / HU_contralateral) × 100%

- NWU < 5%: Lesión aguda (< 3h), reversible
- NWU 5-12%: Subaguda (3-6h), transición
- NWU > 12%: Tardía (> 6h), irreversible
```

Las features firstorder capturan este espectro cuantitativamente.

### 2.3 Familia GLCM (Gray-Level Co-occurrence Matrix)

El **GLCM** analiza **relaciones espaciales entre píxeles vecinos** — textura:

| Feature | Qué mide | Interpretación en stroke |
|---------|----------|---------------------------|
| `glcm_contrast` | Diferencia de intensidad entre píxeles vecinos | Alto contraste = bordes nítidos = lesión bien definida (tardía) |
| `glcm_correlation` | Correlación lineal entre píxeles | Alta correlación = textura homogénea (edema completo) |
| `glcm_homogeneity` (idm) | Diferencias cercanas a diagonal | Alta homogeneidad = cambios graduales (transición penumbra→núcleo) |
| `glcm_energy` (jointenergy) | Uniformidad de la matriz | Energía alta = textura regular (lesión estable) |
| `glcm_entropy` (jointentropy) | Desorden en GLCM | Alta entropía = mezcla compleja de tejidos |
| `glcm_dissimilarity` | Diferencia absoluta promedio | Similar a contraste |
| `glcm_autocorrelation` | Correlación consigo misma | Patrones repetitivos en la lesión |

**Hallazgos visuales correlacionados**:
- **Cinta insular**: GLCM contrast alto en etapas tempranas (pérdida de definición gris/blanco)
- **Núcleo de infarto**: GLCM homogeneity alta, contraste bajo (área homogéneamente hipodensa)
- **Penumbra**: GLCM entropy alta (mezcla de intensidades)

### 2.4 Familia GLSZM (Gray-Level Size Zone Matrix)

El **GLSZM** analiza **zonas de píxeles conectados del mismo nivel de gris** — tamaño de regiones homogéneas:

| Feature | Qué mide | Interpretación en stroke |
|---------|----------|---------------------------|
| `glszm_smallareaemphasis` | Énfasis en zonas pequeñas | Alta = muchas zonas pequeñas = lesión fragmentada/heterogénea |
| `glszm_largeareaemphasis` | Énfasis en zonas grandes | Alta = pocas zonas grandes = lesión consolidada (tardía) |
| `glszm_zonepercentage` | % de zonas vs píxeles | Eficiencia de cobertura |
| `glszm_graylevelvariance` | Varianza de niveles de gris de zonas | Diversidad de intensidades |
| `glszm_sizezonenonuniformity` | No uniformidad en tamaño de zonas | Variabilidad de tamaño de áreas isquémicas |
| `glszm_zonevariance` | Varianza de tamaño de zonas | Heterogeneidad espacial |
| `glszm_smallarealowgraylevelemphasis` | Zonas pequeñas + oscuras | **Núcleo de infarto pequeño** |
| `glszm_largearealowgraylevelemphasis` | Zonas grandes + oscuras | **Núcleo de infarto grande** = más tiempo evolución |
| `glszm_smallareahighgraylevelemphasis` | Zonas pequeñas + brillantes | Penumbra/áreas marginales |
| `glszm_lowgraylevelzoneemphasis` | Énfasis en zonas oscuras | Peso del núcleo isquémico |
| `glszm_highgraylevelzoneemphasis` | Énfasis en zonas brillantes | Peso de tejido salvable |

**Patrones característicos**:

| Etapa | SmallAreaLowGL | LargeAreaLowGL | SmallAreaHighGL |
|-------|---------------|----------------|-----------------|
| **Aguda (< 3h)** | Baja | Muy baja | **Alta** (penumbra predominante) |
| **Subaguda (3-6h)** | Media | Media | Media |
| **Tardía (> 6h)** | Alta (núcleo consolidado) | **Alta** (núcleo grande) | Baja |

---

## 3. El "Reloj de Tejido" vs Reloj Cronológico

### 3.1 Disociación tiempo-biología

```
┌─────────────────────────────────────────────────────────────┐
│  TIEMPO desde síntomas (OTS: Onset-to-Scan)                │
│  └── Variable independiente: reloj del paciente             │
│                                                             │
│  vs                                                         │
│                                                             │
│  EDAD BIOLÓGICA de la lesión (tissue clock)                │
│  └── Variable de interés: grado de daño irreversible       │
└─────────────────────────────────────────────────────────────┘
```

**Problema clínico central**: El tiempo cronológico NO predice bien la reversibilidad porque:

1. **Variabilidad en vulnerabilidad tisular**: Algunos tejidos toleran mejor la isquemia
2. **Colaterales**: Buena circulación colateral = más tiempo de tolerancia
3. **Tamaño del vaso ocluido**: Oclusión distal vs proximal
4. **Temperatura, glucosa, edad**: Factores metabólicos

### 3.2 Lo que ve el modelo CNN-Radiómico

Según el estudio npj Digital Medicine (2024), el modelo CNN-R aprendió a detectar:

```
CNN-R Features → Edad Biológica Estimada
├── Cambios de intensidad (NWU proxy)
├── Patrones texturales (GLCM features)
├── Tamaño y forma de lesión (Shape features)
├── Heterogeneidad interna (GLSZM features)
└── Distribución espacial de HU (FirstOrder features)
```

**Resultado**: R² = 0.58 para edad cronológica, pero mejor correlación con **Core:Penumbra ratio** (ρ² = 0.37) que métodos tradicionales (NWU: ρ² = 0.19).

---

## 4. Recreando el Proceso de Decisión del Neurólogo

### 4.1 Algoritmo mental del neurólogo experto

```
PASO 1: Calidad de TC
├── ¿Ventana apropiada? (WL=35, WW=60 para stroke)
├── ¿Giro de cabeza/artefactos?
└── ¿Corte en nivel correcto? (ganglionar + supraganglionar)

PASO 2: Exclusión de contraindicaciones
├── ¿Hemorragia?
├── ¿Tumor/masas?
└── ¿Infarto antiguo que confunde?

PASO 3: Evaluación ASPECTS
├── ¿Cinta insular preservada? (I)
├── ¿Núcleos basales definidos? (C, L)
├── ¿Cápsula interna visible? (IC)
└── ¿Cortes M1-M6 simétricos?

PASO 4: Estimación de "vida" de la lesión
├── ¿Edema citotóxico completo? (ASPECTS < 7)
├── ¿Signos tempranos sutiles? (ASPECTS 8-10)
└── ¿Arteria hiperdensa? (sugiere oclusión aguda)

PASO 5: Decisión terapéutica
├── ≤ 4.5h + ASPECTS ≥ 6 → Trombólisis IV
├── 4.5-6h + ASPECTS ≥ 6 → Considerar (imaging-guided)
├── > 6h o ASPECTS < 6 → Evaluar perfusión/ventana extendida
└── Desconocido → Usar "tissue clock" (modelo radiómico)
```

### 4.2 Mapeo a features radiómicas

| Paso del neurólogo | Features radiómicas relevantes |
|-------------------|--------------------------------|
| Calidad TC | `firstorder_*` (rango dinámico), `shape2d_*` (artefactos excluidos) |
| Exclusión hemorragia | `firstorder_maximum` (no debe ser > 80 HU) |
| ASPECTS: Cinta insular | `glcm_contrast`, `firstorder_entropy` en región insular |
| ASPECTS: Núcleos basales | `shape2d_*` en ganglios basales |
| Edema completo | `firstorder_mean` muy bajo, `glszm_largearealowgraylevelemphasis` alto |
| Estimación temporal | Combinación de todas las familias (modelo multimodal) |

---

## 5. Implicaciones para el Modelado ML

### 5.1 Features con mayor valor biológico (según literatura)

De los estudios revisados, las features más predictivas:

**Estudio BMC Medical Imaging (2024)** — 9 features seleccionadas por LASSO:
1. Intensidad de la lesión (firstorder)
2. Skewness (asimetría del histograma)
3. Depth-weighted median
4. Depth-weighted interquartile range
5. Depth-weighted skew
6. GLCM correlation
7. Difference features (lesión vs contralateral)
8. Standard deviation depth-weighted
9. Neighborhood Gray-Tone Difference Matrix Busyness

**Estudio npj Digital Medicine (2024)**:
- CNN features + radiómicas hand-crafted
- La combinación CNN-R superó a CNN solo o radiómicas solas
- La **diferencia lesión-contralateral** fue crítica

### 5.2 Estrategia de selección recomendada

```
1. PRE-FILTRADO (robustez)
   ├── ICC ≥ 0.75 (consistencia inter-observador)
   ├── Variance > threshold (no constantes)
   └── No NaN/Inf en > 5% de casos

2. ANÁLISIS UNIVARIADO
   ├── T-test o Mann-Whitney U por feature
   ├── p < 0.05 (no corregido aún)
   └── Effect size (Cliff's δ > 0.5)

3. REDUCCIÓN DE REDUNDANCIA
   ├── Correlación de Pearson |r| < 0.90
   └── mRMR: Maximum Relevance, Minimum Redundancy

4. SELECCIÓN MULTIVARIADO (LASSO)
   ├── Logistic Regression con L1 penalty
   ├── λ óptimo por 10-fold CV
   └── Features con coeficiente ≠ 0

5. VALIDACIÓN
   ├── Bootstrap internal (n=1000)
   ├── Stability analysis (¿cuántas veces aparece cada feature?)
   └── External validation (test set separado)
```

### 5.3 Features a priorizar (hipótesis basada en fisiología)

| Prioridad | Features | Justificación |
|-----------|----------|---------------|
| **Alta** | `firstorder_mean`, `firstorder_median`, `firstorder_minimum` | Directamente relacionadas con NWU |
| **Alta** | `glszm_largearealowgraylevelemphasis` | Tamaño del núcleo irreversible |
| **Alta** | `glcm_contrast`, `glcm_homogeneity` | Definición de bordes de la lesión |
| **Media** | `shape2d_sphericity`, `shape2d_meshsurface` | Morfología de la lesión |
| **Media** | `firstorder_entropy`, `firstorder_skewness` | Heterogeneidad interna |
| **Baja** | `firstorder_kurtosis` | Menos interpretable clínicamente |

---

## 6. Conclusiones para el Modelado

### 6.1 Lo que hemos aprendido

1. **Las radiómicas capturan el "reloj de tejido"** — mejor que el tiempo cronológico para predecir reversibilidad

2. **La combinación CNN + radiómicas es superior** — el deep learning encuentra patrones que los humanos no percibimos

3. **La selección de variables es crítica** — LASSO redujo 2016 features a 9 con mejor rendimiento

4. **El contralateral importa** — features de diferencia lesión/normal son altamente informativas

5. **Shape/FirstOrder/GLCM/GLSZM capturan aspectos distintos** — la combinación es clave

### 6.2 Recomendaciones para nuestro modelo

```
ESTRATEGIA PROPUESTA:

1. Añadir features de "diferencia contralateral"
   ├── Para cada radiómica: (valor_lesión - valor_contralateral) / valor_contralateral
   └── Esto simula el razonamiento del neurólogo (compara lado a lado)

2. Selección agresiva con LASSO
   ├── Iniciar con todas las radiómicas + agregaciones
   ├── LASSO con CV anidado para selección de λ
   └── Mantener solo features seleccionadas en > 80% de folds

3. PCA para reducción dimensional
   ├── No como selección, sino como pre-procesamiento
   ├── 10-20 componentes principales
   └── Interpretar componentes según carga de features

4. Modelos a evaluar
   ├── SVM con kernel RBF (maneja no-linealidad)
   ├── XGBoost (maneja interacciones, robusto)
   ├── LightGBM (eficiente, buen default)
   └── Red neuronal: CNN 1D sobre slices ordenados

5. Arquitectura híbrida (ambiciosa)
   ├── CNN 1D: slices ordenados espacialmente como secuencia
   ├── Autoencoder: reducción no-lineal de features
   └── Ensemble: promedio ponderado de predicciones
```

---

## Referencias

1. Deep learning biomarker of chronometric and biological ischemic stroke lesion age from unenhanced CT. *npj Digital Medicine*, 2024.
2. Non-contrast CT radiomics-clinical machine learning model for futile recanalization. *BMC Medical Imaging*, 2024.
3. Use of the Alberta Stroke Program Early CT Score (ASPECTS). *AJNR*, 2001.
4. Segmenting Ischemic Penumbra and Infarct Core Simultaneously on NCCT. *Biomedicines*, 2024.
5. Predicting the clinical prognosis of acute ischemic stroke using ML. *Frontiers in Neuroinformatics*, 2024.
