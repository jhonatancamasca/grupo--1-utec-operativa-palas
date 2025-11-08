# Dashboard de Rendimiento: Palas y Camiones ⛏️🚛

Dashboard interactivo desarrollado con Streamlit para el análisis operativo de maquinaria pesada en operaciones mineras. Permite visualizar y analizar el rendimiento de palas y camiones mediante múltiples métricas y filtros personalizables.

## 📋 Características

### Análisis de Palas
- **Eficiencia Operativa**: Tonelaje por fase, duración de cargado, eficiencia de carga
- **Análisis de Promedios y Outliers**: Número de pasadas, tiempo de carguío vs payload
- **Análisis por Estación**: Tonelaje y horas trabajadas según temporada (lluviosa/seca)

### Análisis de Camiones
- **Eficiencia Operativa**: Tonelaje por fase, tiempo de viaje, distancia vs tiempo
- **Análisis de Promedios y Outliers**: Pasadas por camión, tiempo de carguío vs payload
- **Análisis Mensual**: Distancia recorrida, cantidad de viajes, tonelaje promedio
- **Análisis por Estación**: Horas trabajadas, distancia promedio, distribución de tonelaje

## 🚀 Instalación

### Requisitos Previos
- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Instalación de Dependencias

```bash
pip install streamlit pandas numpy matplotlib seaborn plotly altair
```

O usando un archivo `requirements.txt`:

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.14.0
altair>=5.0.0
```

## 📁 Estructura de Archivos

```
proyecto/
│
├── dashboard.py              # Archivo principal de la aplicación
├── palas_100k.csv           # Dataset de palas
├── camion_100k.csv          # Dataset de camiones
├── requirements.txt         # Dependencias del proyecto
└── README.md               # Este archivo
```

## 📊 Formato de Datos

### Dataset de Palas (palas_100k.csv)
Columnas requeridas:
- `PRIMARYMACHINENAME`: Identificador de la pala
- `SECONDARYMACHINENAME`: Camión asociado
- `STARTTIME`: Fecha/hora de inicio
- `ENDTIME`: Fecha/hora de fin
- `PAYLOAD`: Tonelaje cargado
- `MATERIALGROUPLEVEL1`: Tipo de material
- `FASE`: Fase o área de extracción
- `LOADINGDURATION`: Duración del cargado (minutos)
- `DIPPERCOUNT`: Número de pasadas

### Dataset de Camiones (camion_100k.csv)
Columnas requeridas:
- `PRIMARYMACHINENAME`: Identificador del camión
- `SECONDARYMACHINENAME`: Pala asociada
- `STARTTIME`: Fecha/hora de inicio
- `ENDTIME`: Fecha/hora de fin
- `PAYLOAD`: Tonelaje transportado
- `MATERIAL`: Tipo de material
- `FASE`: Fase o área de extracción
- `TIEMPO_VIAJE`: Duración del viaje (minutos)
- `TIEMPO_CARGUIO`: Tiempo de carguío (minutos)
- `DISTANCIA_CARGADO_EFH(mts)`: Distancia cargado (metros)
- `DISTANCIA_VACIO_EFH(mts)`: Distancia vacío (metros)
- `DIPPERCOUNT`: Número de pasadas

## 🎮 Uso

### Iniciar la Aplicación

```bash
streamlit run dashboard.py
```

La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`

### Navegación

1. **Seleccionar Máquina**: Elige entre "Palas" o "Camiones" en el menú lateral
2. **Seleccionar Submenu**: Elige el tipo de análisis deseado
3. **Aplicar Filtros**: 
   - Rango de fechas
   - Máquinas específicas
   - Tipo de material
   - Fase de operación
   - Estación del año

### Filtros Disponibles

#### Filtros Temporales
- **Rango de fechas**: Selecciona el período a analizar mediante slider

#### Filtros de Máquinas
- **Palas**: Selección múltiple de palas específicas
- **Camiones**: Selección múltiple de camiones específicos

#### Filtros Operativos
- **Material**: Filtra por tipo de material extraído/transportado
- **Fase**: Filtra por área o fase de extracción
- **Estación**: Filtra por temporada (Lluviosa/Seca)

## 📈 Visualizaciones

### Gráficos de Barras
- Tonelaje por fase y tipo de material
- Horas trabajadas por estación
- Eficiencia de carga

### Gráficos de Dispersión
- Duración de cargado vs tonelaje
- Tiempo de viaje vs distancia total
- Tiempo de carguío vs payload

### Boxplots
- Distribución de número de pasadas
- Distribución de tonelaje por estación

### Gráficos de Líneas
- Evolución mensual de distancia recorrida
- Tendencias de viajes y payload

## 🎨 Paleta de Colores

El dashboard utiliza una paleta de colores consistente:
- **Azul primario**: `#1f77b4`
- **Naranja secundario**: `#ff7f0e`
- **Verde éxito**: `#2ca02c`
- **Rojo advertencia**: `#d62728`
- **Púrpura info**: `#9467bd`

**Estaciones:**
- Lluviosa: `#3498db` (azul)
- Seca: `#f39c12` (naranja)

## ⚙️ Configuración

### Personalizar Colores
Modifica las constantes al inicio del archivo `dashboard.py`:

```python
COLOR_PRIMARY = '#1f77b4'
COLOR_SECONDARY = '#ff7f0e'
PALETA_ESTACIONES = {
    'Lluviosa': '#3498db',
    'Seca': '#f39c12'
}
```

### Ajustar Rutas de Archivos
Modifica las rutas en el código si tus archivos están en otra ubicación:

```python
ruta_palas = "ruta/a/tu/palas_100k.csv"
ruta_camiones = "ruta/a/tu/camion_100k.csv"
```

## 🔧 Troubleshooting

### Error: "File not found"
- Verifica que los archivos CSV estén en la misma carpeta que `dashboard.py`
- Verifica los nombres de los archivos (sensible a mayúsculas/minúsculas)

### Error: "Column not found"
- Asegúrate de que tus CSVs contengan todas las columnas requeridas
- Verifica que los nombres de las columnas coincidan exactamente

### Gráficos no se muestran
- Verifica que los filtros no estén eliminando todos los datos
- Intenta resetear los filtros a sus valores por defecto

### Rendimiento lento
- Reduce el rango de fechas seleccionado
- Limita el número de máquinas seleccionadas
- Considera usar datasets más pequeños para pruebas

## 📝 Notas Importantes

- Los datos de material "unknown" son automáticamente excluidos del análisis
- Las estaciones están configuradas para la región de Junín, Perú
- Los gráficos se generan dinámicamente según los filtros aplicados
- La aplicación usa caché de Streamlit para mejorar el rendimiento

## 🤝 Contribuciones

Para contribuir al proyecto:
1. Asegúrate de que el código siga las convenciones de estilo de Python (PEP 8)
2. Documenta cualquier nueva funcionalidad
3. Prueba exhaustivamente antes de hacer cambios

## 📄 Licencia

Este proyecto está desarrollado para análisis interno de operaciones mineras.

## 👥 Soporte

Para reportar problemas o sugerir mejoras, contacta al equipo de desarrollo.

---

**Versión**: 1.0  
**Última actualización**: Noviembre 2025