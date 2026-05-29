# Data Dictionary — `acv_clasif_train.csv` / `acv_clasif_test.csv`

> Documento vivo. Cualquier descubrimiento durante EDA que contradiga este diccionario debe
> actualizarlo. La fuente de verdad para limpieza y *staging* es esta tabla.
>
> Última revisión: 2026-05-25 (Fase 1 en curso).

## Información general

| Atributo | Valor |
|---|---|
| Origen físico | CSV planos exportados desde el flujo de PyRadiomics + HIS de FVL |
| Encoding | UTF-8 |
| Delimitador | `,` (coma); todos los strings entre comillas dobles |
| Grano físico | **1 fila = 1 slice de TC craneal** de un paciente |
| Grano analítico | **1 paciente = N slices** (agregar para ML y KPIs) |
| Filas train | 1,998 |
| Filas test | 822 |
| Pacientes únicos train | 55 |
| Pacientes únicos test | 19 |
| Solapamiento pacientes train↔test | 0 (split a nivel paciente — correcto) |

## Convenciones

- `family` = familia semántica (shape, firstorder, glcm, glszm, demo, clinical, target, key).
- `nullable` = si la fuente actual contiene celdas vacías (`""`) o `NaN`.
- `pii` = si la columna es un identificador potencialmente sensible.
- `use_in_model` = si la columna entra al modelo *as-is*. **Cualquier columna marcada `leakage` no se usa nunca como feature.**

## Columnas

| # | Columna (raw) | Renombrado en staging | Tipo | family | Rango / valores | nullable | pii | use_in_model | Notas |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `ID` | `patient_id` | string | key | identificador opaco | no | sí | no | **clave de agrupación**. Se repite porque un paciente tiene N slices. Hashear si el repo se hace público. |
| 2 | `original_shape2D_MeshSurface` | `shape2d_meshsurface` | float | shape | ≥ 0 | no | no | sí | área aprox. de la lesión 2D |
| 3 | `original_shape2D_Perimeter` | `shape2d_perimeter` | float | shape | ≥ 0 | no | no | sí | perímetro 2D |
| 4 | `original_shape2D_Sphericity` | `shape2d_sphericity` | float | shape | [0, 1] | no | no | sí | 1 = círculo perfecto |
| 5 | `original_shape2D_SphericalDisproportion` | `shape2d_sphericaldisproportion` | float | shape | ≥ 1 | no | no | sí | inverso de esfericidad |
| 6 | `original_shape2D_MaximumDiameter` | `shape2d_maximumdiameter` | float | shape | ≥ 0 | no | no | sí | diámetro mayor |
| 7 | `original_shape2D_MajorAxisLength` | `shape2d_majoraxislength` | float | shape | ≥ 0 | no | no | sí | eje mayor del elipsoide envolvente |
| 8 | `original_shape2D_MinorAxisLength` | `shape2d_minoraxislength` | float | shape | ≥ 0 | no | no | sí | eje menor del elipsoide envolvente |
| 9 | `original_shape2D_Elongation` | `shape2d_elongation` | float | shape | [0, 1] | no | no | sí | 0 = línea, 1 = círculo |
| 10–27 | `original_firstorder_*` (18 cols) | `firstorder_<name>` | float | firstorder | varía por feature | no | no | sí | estadísticos del histograma de intensidades |
| 28–51 | `original_glcm_*` (24 cols) | `glcm_<name>` | float | glcm | varía por feature | no | no | sí | matriz de co-ocurrencia (textura) |
| 52–67 | `original_glszm_*` (16 cols) | `glszm_<name>` | float | glszm | varía por feature | no | no | sí | matriz de zonas de tamaño (textura) |
| 68 | `Patien Age in Study` | `age_years` | int | demo | [0, 120] | no | no | sí | **typo en fuente** ("Patien" en vez de "Patient"). Se preserva en `raw`, se renombra en `staging`. |
| 69 | `Patient Sex` | `sex` | enum | demo | `M` / `F` | no | no | sí | drift de sexo entre train (61% F) y test (64% M) — ver EDA |
| 70 | `NIHSS` | `nihss` | float | clinical | [0, 42] ∪ NULL | sí (train ~10%, test 0%) | no | sí (+ missing indicator) | escala de severidad neurológica; faltante = MAR clínico |
| 71 | `Evolution Time` | `evolution_minutes` (+ derivada `evolution_hours = minutes/60`) | float | clinical | ≥ 0 (minutos) | no | no | **NO — leakage** | **MINUTOS** desde inicio de síntomas (confirmado por EDA: cutoff 270 min reproduce target 100%). Es la base aritmética del target; usarla como feature es leakage directo |
| 72 | `ASPECTS` | `aspects` | int | clinical | [0, 10] ∪ NULL | sí (train ~8%, test ~19%) | no | sí (+ missing indicator) | score CT de afectación isquémica; 10 = normal |
| 73 | `Evolution Time_Clas` | `is_over_window` | int | target | {0, 1} | no | no | target | 0 = ≤ 4.5 h (elegible para trombólisis), 1 = > 4.5 h |

## Notas semánticas críticas

1. **`ID` no es PK.** Es clave natural de paciente. La PK de la fila (slice) es `(ID, slice_order)` y `slice_order` no viene en la fuente — se sintetiza en `staging` como `row_number() OVER (PARTITION BY ID ORDER BY <natural order del CSV>)`.

2. **`Evolution Time` ↔ `Evolution Time_Clas`.** Por definición operacional confirmada en EDA: `Evolution Time_Clas = 1 si Evolution Time > 270` (con la columna en **minutos**). El cutoff equivalente clínico es 4.5 h = 270 min. Esto se valida en Great Expectations (`patient_integrity` suite). Usar `Evolution Time` como feature destruye el modelo (leakage).

3. **NIHSS faltante en train pero no en test.** Posible patrón sistémico: ¿el equipo que generó test imputó? Hay que auditar. En cualquier caso, en train se debe modelar con missing indicator.

4. **ASPECTS faltante en test (~19%)** es mayor que en train (~8%). Indica posible diferencia de protocolo entre cohortes. Auditar drift, no asumir distribuciones iguales.

5. **Familias radiómicas** se conservan agrupadas en staging para facilitar selección por familia en feature engineering.

## Próximas validaciones (Great Expectations — Fase 3)

- `expect_column_values_to_be_between` para todas las features radiómicas con rangos clínicamente plausibles (a confirmar contra documentación de PyRadiomics).
- `expect_column_pair_values_A_to_be_greater_than_B` para `evolution_minutes > 270` ↔ `is_over_window = 1` (100% match esperado).
- `expect_column_values_to_be_in_set` para `sex ∈ {M, F}`, `is_over_window ∈ {0,1}`, `aspects ∈ [0,10]`.
- `expect_table_row_count_to_be_between` para detectar cargas truncadas.
- `expect_compound_columns_to_be_unique` para `(patient_id, slice_order)`.
