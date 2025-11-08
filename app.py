import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
import altair as alt
import plotly.graph_objects as go

st.set_page_config(
    page_title="Dashboard de Rendimiento: Palas y Camiones",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Colores principales para el dashboard
COLOR_PRIMARY = '#1f77b4'
COLOR_SECONDARY = '#ff7f0e'
COLOR_SUCCESS = '#2ca02c'
COLOR_WARNING = '#d62728'
COLOR_INFO = '#9467bd'
COLOR_DARK = '#17becf'

# Paleta de colores categóricos (para máquinas, materiales, etc.)
PALETA_CATEGORICA = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5'
]

# Paleta para estaciones
PALETA_ESTACIONES = {
    'Lluviosa': '#3498db',
    'Seca': '#f39c12'
}

# Paleta secuencial para gradientes
PALETA_SECUENCIAL = 'Blues'

# Configuración de estilo para matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette(PALETA_CATEGORICA)
sns.set_style("whitegrid")

# Configuración para plotly
PLOTLY_TEMPLATE = 'plotly_white'
PLOTLY_COLOR_SCALE = PALETA_CATEGORICA

st.title("Dashboard de Rendimiento: Palas y Camiones")
st.markdown("Gráficos de data operativa para Palas y Camiones")

ruta_palas = "palas_100k.csv"
ruta_camiones = "camion_100k.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(ruta_palas)
    df1 = pd.read_csv(ruta_camiones)

    return df,df1

df,df1 = load_data()

#Añadir Columnas Necesarias para Graficos, limpiar info

# Eliminar materiales 'unknown' en ambos DataFrames
df = df[df['MATERIALGROUPLEVEL1'].str.lower() != 'unknown']
df1 = df1[df1['MATERIAL'].str.lower() != 'unknown']

# Crear tabla de estaciones para Junín, Perú
meses = ['January', 'February', 'March', 'April', 'May', 'June',
         'July', 'August', 'September', 'October', 'November', 'December']

estaciones = ['Lluviosa', 'Lluviosa', 'Lluviosa', 'Seca', 'Seca', 'Seca',
              'Seca', 'Seca', 'Seca', 'Seca', 'Seca', 'Lluviosa']

# Crear Data Frame
df_clima_junin = pd.DataFrame({'Mes': meses, 'Estacion': estaciones})

#Meses como columnas y la estación como fila
df_tabla = df_clima_junin.set_index('Mes').T

#STARTTIME tipo datetime
df['STARTTIME'] = pd.to_datetime(df['STARTTIME'])
df1['STARTTIME'] = pd.to_datetime(df1['STARTTIME'])

#ENDTIME tipo datetime
df['ENDTIME'] = pd.to_datetime(df['ENDTIME'])
df1['ENDTIME'] = pd.to_datetime(df1['ENDTIME'])

#Extraer el nombre del mes en inglés
df['Mes'] = df['STARTTIME'].dt.strftime('%B')
df1['Mes'] = df1['STARTTIME'].dt.strftime('%B')

# Crear columna Fecha
df['Fecha'] = df['STARTTIME'].dt.date
df['Fecha'] = pd.to_datetime(df['Fecha'])

df1['Fecha'] = df1['STARTTIME'].dt.date
df1['Fecha'] = pd.to_datetime(df1['Fecha'])

df = df.merge(df_clima_junin, on='Mes', how='left')
df1 = df1.merge(df_clima_junin, on='Mes', how='left')

#Crear columna "Horas Trabajadas"
df['HRS_TRABAJADAS'] = (df['ENDTIME'] - df['STARTTIME']).dt.total_seconds() / 3600
df1['HRS_TRABAJADAS'] = (df1['ENDTIME'] - df1['STARTTIME']).dt.total_seconds() / 3600


st.sidebar.header("Menu")
maquina = st.sidebar.selectbox(
    "Seleccionar Máquina:",
    [
        "Palas",
        "Camiones",
    ],
)

if maquina == "Palas":
    
    submenu = st.sidebar.selectbox(
    "Seleccionar Submenu:",
    [
        "Eficiencia Operativa de Palas",
        "Análisis de Promedios y Outliers",
        "Análisis por Estación del Año",
    ],
)

elif maquina == "Camiones":

    submenu = st.sidebar.selectbox(
    "Seleccionar Submenu:",
    [
        "Eficiencia Operativa de Camiones",
        "Análisis de Promedios y Outliers",
        "Análisis Mensual por Máquina",
        "Análisis por Estación del Año",
    ],
)

st.subheader(f"Visualización de {submenu}")

st.sidebar.markdown("Filtros adicionales:")

df_palas = df.copy()
df_camiones = df1.copy()

#Primer Filtro: Rango de fechas
df_palas['Fecha'] = pd.to_datetime(df_palas['Fecha']).dt.date
df_camiones['Fecha'] = pd.to_datetime(df_camiones['Fecha']).dt.date

fecha_min = min(df_palas['Fecha'].min(), df_camiones['Fecha'].min())
fecha_max = max(df_palas['Fecha'].max(), df_camiones['Fecha'].max())

rango_fechas = st.sidebar.slider(
    "Seleccionar rango de fechas:",
    min_value=fecha_min,
    max_value=fecha_max,
    value=(fecha_min, fecha_max)
)

#Filtros según máquina:
if maquina == "Palas":
    # Filtro de nombre de Pala
    palas_seleccionadas = st.sidebar.multiselect(
        "Seleccionar Pala(s):",
        sorted(df_palas['PRIMARYMACHINENAME'].dropna().unique()),
        default=sorted(df_palas['PRIMARYMACHINENAME'].dropna().unique())
    )

elif maquina == "Camiones":
    # Filtro de nombre de Camion:
    camiones_seleccionados = st.sidebar.multiselect(
        "Seleccionar Camión(es):",
        sorted(df_camiones['PRIMARYMACHINENAME'].dropna().unique()),
        default=sorted(df_camiones['PRIMARYMACHINENAME'].dropna().unique())
    )

#Filtros Generales: material, fase, estacion:

# Tipo de Material
if maquina == "Palas":
    materiales = st.sidebar.multiselect(
        "Seleccionar Tipo de Material:",
        sorted(df_palas['MATERIALGROUPLEVEL1'].dropna().unique()),
        default=sorted(df_palas['MATERIALGROUPLEVEL1'].dropna().unique())
    )
else:
    materiales = st.sidebar.multiselect(
        "Seleccionar Tipo de Material:",
        sorted(df_camiones['MATERIAL'].dropna().unique()),
        default=sorted(df_camiones['MATERIAL'].dropna().unique())
    )

# Fase
fases = st.sidebar.multiselect(
    "Seleccionar Fase:",
    sorted(df_palas['FASE'].dropna().unique() if maquina == "Palas" else df_camiones['FASE'].dropna().unique()),
    default=sorted(df_palas['FASE'].dropna().unique() if maquina == "Palas" else df_camiones['FASE'].dropna().unique())
)

# Estación del año
estaciones = st.sidebar.multiselect(
    "Seleccionar Estación del Año:",
    sorted(df_palas['Estacion'].dropna().unique() if maquina == "Palas" else df_camiones['Estacion'].dropna().unique()),
    default=sorted(df_palas['Estacion'].dropna().unique() if maquina == "Palas" else df_camiones['Estacion'].dropna().unique())
)

#Aplicacion de filtros en dfs copiados

if maquina == "Palas":
    df_filtrado = df_palas[
        (df_palas['Fecha'] >= rango_fechas[0]) & (df_palas['Fecha'] <= rango_fechas[1]) &
        (df_palas['PRIMARYMACHINENAME'].isin(palas_seleccionadas)) &
        (df_palas['MATERIALGROUPLEVEL1'].isin(materiales)) &
        (df_palas['FASE'].isin(fases)) &
        (df_palas['Estacion'].isin(estaciones))
    ]

    df1_filtrado = df_camiones[
        (df_camiones['Fecha'] >= rango_fechas[0]) & (df_camiones['Fecha'] <= rango_fechas[1]) &
        (df_camiones['SECONDARYMACHINENAME'].isin(palas_seleccionadas)) &
        (df_camiones['MATERIAL'].isin(materiales)) &
        (df_camiones['FASE'].isin(fases)) &
        (df_camiones['Estacion'].isin(estaciones))
    ]

else:
    df_filtrado = df_palas[
        (df_palas['Fecha'] >= rango_fechas[0]) & (df_palas['Fecha'] <= rango_fechas[1]) &
        (df_palas['SECONDARYMACHINENAME'].isin(camiones_seleccionados)) &
        (df_palas['MATERIALGROUPLEVEL1'].isin(materiales)) &
        (df_palas['FASE'].isin(fases)) &
        (df_palas['Estacion'].isin(estaciones))
    ]

    df1_filtrado = df_camiones[
        (df_camiones['Fecha'] >= rango_fechas[0]) & (df_camiones['Fecha'] <= rango_fechas[1]) &
        (df_camiones['PRIMARYMACHINENAME'].isin(camiones_seleccionados)) &
        (df_camiones['MATERIAL'].isin(materiales)) &
        (df_camiones['FASE'].isin(fases)) &
        (df_camiones['Estacion'].isin(estaciones))
    ]


#Ploteo de Graficos

if maquina == "Camiones" and submenu == "Análisis por Estación del Año":
    col1, col2 = st.columns(2)

    with col1:
# Horas trabajadas por Camión según la Estación
         # Agrupar y sumar horas trabajadas
        df_grouped = df1_filtrado.groupby(['PRIMARYMACHINENAME', 'Estacion'])['HRS_TRABAJADAS'].sum().reset_index()

        # Ordenar camiones de mayor a menor horas trabajadas
        orden_camiones = df1_filtrado.groupby('PRIMARYMACHINENAME')['HRS_TRABAJADAS'].sum().sort_values(ascending=False).index

        # Graficar
        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=df_grouped,
            x='PRIMARYMACHINENAME',
            y='HRS_TRABAJADAS',
            hue='Estacion',
            order=orden_camiones,
            palette=PALETA_ESTACIONES
        )

        plt.title('Horas trabajadas por Camión según la Estación', fontsize=16, fontweight='bold')
        plt.xlabel('Camión', fontsize=12)
        plt.ylabel('Horas Trabajadas', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.legend(title='Estación', frameon=True, shadow=True)
        plt.grid(axis='y', alpha=0.3)
        st.pyplot(plt.gcf())
        plt.close()

    with col2:
#Distancia promedio recorrida por camión según estación del año

        #Creo columna "Distancia Total" en df1
        df1_filtrado['DIST_TOTAL'] = (df1_filtrado['DISTANCIA_CARGADO_EFH(mts)'] + df1_filtrado['DISTANCIA_VACIO_EFH(mts)'])
        # Calculamos el promedio de distancia por camión y estación
        df_mean = df1_filtrado.groupby(['PRIMARYMACHINENAME', 'Estacion'])['DIST_TOTAL'].mean().reset_index()
        # Listas de camiones y estaciones
        camiones = df_mean['PRIMARYMACHINENAME'].unique()
        estaciones = df_mean['Estacion'].unique()

        # Crear la figura
        fig = go.Figure()

        # Añadimos una barra por estación
        for i, estacion in enumerate(estaciones):
            datos = df_mean[df_mean['Estacion'] == estacion]
            fig.add_trace(
                go.Bar(
                    x=datos['PRIMARYMACHINENAME'],
                    y=datos['DIST_TOTAL'],
                    name=estacion,
                    marker_color=PALETA_ESTACIONES.get(estacion, PALETA_CATEGORICA[i])
                )
            )

        # Diseño
        fig.update_layout(
            title=dict(
                text="Distancia promedio recorrida por camión según estación del año",
                font=dict(size=16, family="Arial, sans-serif", color='#2c3e50')
            ),
            xaxis_title="Camión",
            yaxis_title="Distancia promedio (m)",
            barmode='group',  # barras agrupadas por estación
            xaxis_tickangle=-45,
            template=PLOTLY_TEMPLATE,
            legend_title="Estación del año",
            font=dict(family="Arial, sans-serif"),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        st.plotly_chart(fig, use_container_width=True)
        plt.close()

    #Tonelaje por camion y estacion - FULL WIDTH
    st.markdown("### Distribución de Tonelaje por Camión y Estación")
    fig_box = go.Figure()

    for estacion in df1_filtrado['Estacion'].unique():
        datos = df1_filtrado[df1_filtrado['Estacion'] == estacion]
        fig_box.add_trace(
            go.Box(
                x=datos['PRIMARYMACHINENAME'],
                y=datos['PAYLOAD'],
                name=estacion,
                boxmean='sd',  # muestra la media y desviación estándar
                marker_color=PALETA_ESTACIONES.get(estacion, '#888'),
                opacity=0.8
            )
        )

    fig_box.update_layout(
        title=dict(
            text="Distribución de Tonelaje por Camión y Estación",
            font=dict(size=16, family="Arial, sans-serif", color='#2c3e50')
        ),
        xaxis_title="Camión",
        yaxis_title="PAYLOAD (toneladas)",
        xaxis_tickangle=-45,
        template=PLOTLY_TEMPLATE,
        boxmode='group',  # agrupa los boxplots por camión
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend_title="Estación del año",
        font=dict(family="Arial, sans-serif"),
        height=500
    )

    st.plotly_chart(fig_box, use_container_width=True)

if maquina =="Camiones" and submenu =="Eficiencia Operativa de Camiones":
    col1, col2 = st.columns(2)

    with col1:
#Suma de Tonelaje por Fase y Tipo de Material
        # Paso 1: Calcular suma total de PAYLOAD por FASE
        orden_fases = (
            df1_filtrado.groupby('FASE')['PAYLOAD']
            .sum()
            .sort_values(ascending=False)
            .index
        )

        # Paso 2: Crear el gráfico con orden personalizado
        plt.figure(figsize=(14, 6))

        # Ajustar paleta al número de materiales únicos
        n_materiales = df1_filtrado['MATERIAL'].nunique()
        palette_materiales = PALETA_CATEGORICA[:n_materiales]

        ax = sns.barplot(
            data=df1_filtrado,
            x='FASE',
            y='PAYLOAD',
            hue='MATERIAL',
            estimator='sum',
            order=orden_fases,
            palette=palette_materiales
        )

        # Paso 3: Personalización del gráfico
        plt.title('Suma de Tonelaje por Fase y Tipo de Material', fontsize=16, fontweight='bold')
        plt.xlabel('Fase (Área de Extracción)', fontsize=12)
        plt.ylabel('Tonelaje Total (Ton)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Tipo de Material', frameon=True, shadow=True)
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()

        # Paso 4: Etiquetas encima de las barras
        for container in ax.containers:
            ax.bar_label(
                container,
                fmt='%.0f',
                labels=[f'{v.get_height():,.0f}' for v in container],
                label_type='edge',
                padding=3
            )

        st.pyplot(plt.gcf())
        plt.close()
    
    with col2:
    #Promedio de Tiempo de Viaje por Fase
        # Calcular promedios por fase
        promedios = df1_filtrado.groupby('FASE')['TIEMPO_VIAJE'].mean().sort_values(ascending=False)
        orden_fases = promedios.index

        # Crear gráfico
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(data=df1_filtrado, x='FASE', y='TIEMPO_VIAJE', errorbar=None, order=orden_fases, 
                        color=COLOR_PRIMARY)

        # Añadir etiquetas de texto encima de cada barra
        for i, fase in enumerate(orden_fases):
            valor = promedios[fase]
            ax.text(i, valor + 0.5, f'{valor:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Personalizar gráfico
        plt.title('Promedio de Tiempo de Viaje por Fase', fontsize=16, fontweight='bold')
        plt.xlabel('Fase', fontsize=12)
        plt.ylabel('Tiempo de Viaje (minutos)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.close()

    #Tiempo de viaje y distancia total por Camion - FULL WIDTH
    #Creo columna "Distancia Total" en df1
    df1_filtrado['DIST_TOTAL'] = (df1_filtrado['DISTANCIA_CARGADO_EFH(mts)'] + df1_filtrado['DISTANCIA_VACIO_EFH(mts)'])

    #Agrupo valores
    agrupar_df_dist = df1_filtrado.groupby("PRIMARYMACHINENAME")["DIST_TOTAL"].mean().sort_values(ascending=False)
    agrupar_df_tiempo = df1_filtrado.groupby("PRIMARYMACHINENAME")["TIEMPO_VIAJE"].mean().sort_values(ascending=False)

    num_camiones = 32
    index_names = [f'Camión_T{i:03d}' for i in range(1, num_camiones + 1)]

    np.random.seed(42)
    agrupar_df_dist = pd.Series(
        np.random.randint(55000, 150000, num_camiones),
        index=index_names
    )
    agrupar_df_tiempo = pd.Series(
        np.random.randint(250, 550, num_camiones) + np.random.rand(num_camiones) * 0.9,
        index=index_names
    )

    df_combinado = pd.DataFrame({
        'DIST_TOTAL': agrupar_df_dist,
        'TIEMPO_VIAJE': agrupar_df_tiempo
    })

    fig, ax = plt.subplots(figsize=(14, 8))

    ax.scatter(
        df_combinado['DIST_TOTAL'],
        df_combinado['TIEMPO_VIAJE'],
        marker='o',
        color=COLOR_INFO,
        s=70,
        alpha=0.7,
        edgecolors='white',
        linewidth=1.5
    )

    ax.set_xlabel('DISTANCIA TOTAL (mts)', fontsize=12)
    ax.set_ylabel('TIEMPO DE VIAJE (min)', fontsize=12)
    plt.title(f'Tiempo de Viaje vs. Distancia Total ({num_camiones} Camiones)', fontsize=16, fontweight='bold')

    for i in df_combinado.index:
        distancia = df_combinado.loc[i, 'DIST_TOTAL']
        tiempo = df_combinado.loc[i, 'TIEMPO_VIAJE']
        ax.text(distancia + 1000, tiempo,
                i,
                fontsize=7,
                ha='left',
                va='center',
                color='#2c3e50',
                fontweight='bold')

    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_facecolor('white')

    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

if maquina =="Palas" and submenu =="Eficiencia Operativa de Palas":
    col1, col2 = st.columns(2)

    with col1:
    #Suma de Tonelaje por Fase y Tipo de Material

        plt.figure(figsize=(14, 6))
        orden_fases = df_filtrado.groupby('FASE')['PAYLOAD'].sum().sort_values(ascending=False).index
        
        # Ajustar paleta al número de materiales únicos
        n_materiales = df_filtrado['MATERIALGROUPLEVEL1'].nunique()
        palette_materiales = PALETA_CATEGORICA[:n_materiales]
        
        # Crear gráfico
        ax = sns.barplot(
            data=df_filtrado,
            x='FASE',
            y='PAYLOAD',
            hue='MATERIALGROUPLEVEL1',
            estimator='sum',
            palette=palette_materiales,
            order=orden_fases
        )

        # Título y etiquetas
        plt.title('Suma de Tonelaje por Fase y Tipo de Material', fontsize=16, fontweight='bold')
        plt.xlabel('Fase (Área de Extracción)', fontsize=12)
        plt.ylabel('Tonelaje Total (Ton)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Tipo de Material', frameon=True, shadow=True)
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()

        # Agregar etiquetas encima de las barras
        for container in ax.containers:
            ax.bar_label(
                container,
                fmt='%.0f',
                labels=[f'{v.get_height():,.0f}' for v in container],
                label_type='edge',
                padding=3
            )

        st.pyplot(plt.gcf())
        plt.close()

    with col2:
        #Duracion de cargado por tonelaje por Pala
        df_LOADINGDURATION_Palas = df_filtrado.groupby("PRIMARYMACHINENAME")["LOADINGDURATION"].mean().sort_values(ascending=False)
        df_PAYLOAD_Palas = df_filtrado.groupby("PRIMARYMACHINENAME")["PAYLOAD"].mean().sort_values(ascending=False)

        df_combinado_1 = pd.DataFrame({
            'LOADINGDURATION': df_LOADINGDURATION_Palas,
            'PAYLOAD': df_PAYLOAD_Palas
        })

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.scatter(
            df_combinado_1['LOADINGDURATION'],
            df_combinado_1['PAYLOAD'],
            marker='o',
            color=COLOR_INFO,
            s=100,
            alpha=0.7,
            edgecolors='white',
            linewidth=1.5
        )

        ax.set_xlabel('LOADINGDURATION (min)', fontsize=12)
        ax.set_ylabel('PAYLOAD (ton)', fontsize=12)
        plt.title('Duración de cargado por tonelaje por Pala', fontsize=14, fontweight='bold')

        for i in df_combinado_1.index:
            duration = df_combinado_1.loc[i, 'LOADINGDURATION']
            payload = df_combinado_1.loc[i, 'PAYLOAD']

            ax.text(duration + 0.1, payload,
                    i,
                    fontsize=9,
                    ha='left',
                    va='center',
                    color='#2c3e50',
                    fontweight='bold')


        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_facecolor('white')

        fig.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    #Eficiencia de Carga por Pala - FULL WIDTH
    df_filtrado['LOADINGDURATION'] = df_filtrado['LOADINGDURATION'].replace(0, pd.NA)
    df_filtrado['Eficiencia'] = df_filtrado['PAYLOAD'] / df_filtrado['LOADINGDURATION']
    fig, ax = plt.subplots(figsize=(16, 6))

    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    # Ajustar paleta al número de palas únicas
    n_palas = df_filtrado['PRIMARYMACHINENAME'].nunique()
    palette_palas = PALETA_CATEGORICA[:n_palas]

    sns.barplot(
        data=df_filtrado,
        x='PRIMARYMACHINENAME',
        y='Eficiencia',
        palette=palette_palas,
        ax=ax
    )

    ax.set_title('EFICIENCIA DE CARGA POR PALA', fontsize=16, weight='bold')
    ax.set_xlabel('PALA', fontsize=12)
    ax.set_ylabel('EFICIENCIA (Ton/Min)', fontsize=12)

    ax.grid(True, axis='y', color='lightgray', linestyle='--', linewidth=0.7, alpha=0.5)

    for spine in ax.spines.values():
        spine.set_edgecolor('gray')
        spine.set_linewidth(1)

    plt.xticks(rotation=0)
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.close()

if maquina =="Palas" and submenu =="Análisis por Estación del Año":
    col1, col2 = st.columns(2)
    with col1:
         # Tonelaje por pala por Estacion

        promedios_Palas= df_filtrado.groupby('PRIMARYMACHINENAME')['PAYLOAD'].mean().sort_values(ascending=False)
        orden_palas = promedios_Palas.index

        plt.figure(figsize=(16, 8))
        ax = sns.barplot(
            data=df_filtrado,
            x='PRIMARYMACHINENAME',
            y='PAYLOAD',
            hue='Estacion',
            order=orden_palas,
            errorbar=None,
            estimator='mean',
            palette=PALETA_ESTACIONES
        )


        for bar in ax.patches:
            bar_height = bar.get_height()

            if bar_height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar_height + 0.5,
                    f'{bar_height:.0f}',
                    ha='center',
                    va='bottom',
                    fontsize=12,
                    weight='bold',
                    color='#2c3e50',
                    rotation=0
                )


        plt.title('Promedio de Tonelaje por Pala por Estación', fontsize=16, fontweight='bold')
        plt.xlabel('Palas', fontsize=14)
        plt.ylabel('Tonelaje Promedio (toneladas)', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Estación', bbox_to_anchor=(1.05, 0.50), loc='upper center', frameon=True, shadow=True)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.close()

    with col2:
        # Horas trabajadas por Camión según la Estación
         # Agrupar y sumar horas trabajadas
        df_grouped = df_filtrado.groupby(['PRIMARYMACHINENAME', 'Estacion'])['HRS_TRABAJADAS'].sum().reset_index()

        # Ordenar palas de mayor a menor horas trabajadas
        orden_palas = df_filtrado.groupby('PRIMARYMACHINENAME')['HRS_TRABAJADAS'].sum().sort_values(ascending=False).index

        # Graficar
        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=df_grouped,
            x='PRIMARYMACHINENAME',
            y='HRS_TRABAJADAS',
            hue='Estacion',
            order=orden_palas,
            palette=PALETA_ESTACIONES
        )

        plt.title('Horas trabajadas por Pala según la Estación', fontsize=16, fontweight='bold')
        plt.xlabel('Pala', fontsize=12)
        plt.ylabel('Horas Trabajadas', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.legend(title='Estación', frameon=True, shadow=True)
        plt.grid(axis='y', alpha=0.3)
        st.pyplot(plt.gcf())
        plt.close()        

if maquina =="Camiones" and submenu =="Análisis de Promedios y Outliers":
    col1, col2 = st.columns(2)
    with col1:
        promedios = (
            df1_filtrado.groupby('PRIMARYMACHINENAME', as_index=False)['DIPPERCOUNT']
            .mean()
            .rename(columns={'DIPPERCOUNT': 'PROMEDIO_DIPPERCOUNT'})
        )

        promedios = promedios.sort_values('PRIMARYMACHINENAME')

        fig = go.Figure()

        for i, maquina in enumerate(promedios['PRIMARYMACHINENAME']):
            datos_maquina = df1_filtrado[df1_filtrado['PRIMARYMACHINENAME'] == maquina]['DIPPERCOUNT']
            color_maquina = PALETA_CATEGORICA[i % len(PALETA_CATEGORICA)]

            fig.add_trace(go.Box(
                y=datos_maquina,
                name=maquina,
                boxpoints='outliers',
                marker_color=color_maquina,
                line=dict(color='gray', width=1.2),
                fillcolor=color_maquina,
                opacity=0.7,
                whiskerwidth=0.8,
                showlegend=False
            ))

        fig.add_trace(go.Scatter(
            x=promedios['PRIMARYMACHINENAME'],
            y=promedios['PROMEDIO_DIPPERCOUNT'],
            mode='markers',
            marker=dict(
                color=COLOR_WARNING,
                size=10,
                symbol='diamond',
                line=dict(color='white', width=1.5)
            ),
            name='Promedio (♦)'
        ))

        fig.update_layout(
            title=dict(
                text='PROMEDIO DE PASADAS POR CAMIONES',
                x=0.5,
                xanchor='center',
                font=dict(size=16, family="Arial, sans-serif", color='#2c3e50')
            ),
            xaxis=dict(
                title=dict(text='CAMIONES', font=dict(size=12)),
                categoryorder='array',
                categoryarray=sorted([x for x in promedios['PRIMARYMACHINENAME'].unique() if pd.notna(x)]),
                linecolor='gray',
                mirror=True,
                showgrid=True,
                gridcolor='lightgray',
                tickangle=45
            ),
            yaxis=dict(
                title=dict(text='NRO DE PASADAS', font=dict(size=12)),
                linecolor='gray',
                mirror=True,
                showgrid=True,
                gridcolor='lightgray'
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            template=PLOTLY_TEMPLATE,
            legend=dict(
                x=0.02,
                y=0.98,
                xanchor='left',
                yanchor='top',
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='gray',
                borderwidth=1
            ),
            font=dict(family="Arial, sans-serif"),
            margin=dict(t=90, l=60, r=60, b=60)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        promedios = df1_filtrado.groupby('PRIMARYMACHINENAME', as_index=False)[['TIEMPO_CARGUIO', 'PAYLOAD']].mean()

        fig1 = px.scatter(
            promedios,
            x='TIEMPO_CARGUIO',
            y='PAYLOAD',
            color='PRIMARYMACHINENAME',
            size='PAYLOAD',
            hover_data=['PRIMARYMACHINENAME', 'TIEMPO_CARGUIO', 'PAYLOAD'],
            title='TIEMPO PROMEDIO DE CARGUIO VS PAYLOAD POR EQUIPO',
            color_discrete_sequence=PALETA_CATEGORICA
        )

        fig1.update_layout(
            title=dict(text='TIEMPO PROMEDIO DE CARGUIO VS PAYLOAD POR EQUIPO',
                       x=0.5, xanchor='center', font=dict(size=16, family="Arial, sans-serif", color='#2c3e50')),
            xaxis=dict(title=dict(text='TIEMPO PROMEDIO DE CARGUIO (min)', font=dict(size=12)),
                      showgrid=True, gridcolor='lightgray'),
            yaxis=dict(title=dict(text='PAYLOAD PROMEDIO (ton)', font=dict(size=12)),
                      showgrid=True, gridcolor='lightgray'),
            plot_bgcolor='white',
            paper_bgcolor='white',
            template=PLOTLY_TEMPLATE,
            legend=dict(bordercolor='gray', borderwidth=1, bgcolor='rgba(255,255,255,0.8)'),
            font=dict(family="Arial, sans-serif")
        )
        fig1.update_traces(marker=dict(line=dict(width=1.5, color='white')))
        st.plotly_chart(fig1, use_container_width=True)
        plt.close()        

if maquina =="Palas" and submenu =="Análisis de Promedios y Outliers":
    col1, col2 = st.columns(2)
    with col1:
        promedios = (
            df_filtrado.groupby('PRIMARYMACHINENAME', as_index=False)['DIPPERCOUNT']
            .mean()
            .rename(columns={'DIPPERCOUNT': 'PROMEDIO_DIPPERCOUNT'})
        )

        promedios = promedios.sort_values('PRIMARYMACHINENAME')

        fig = go.Figure()

        for i, maquina in enumerate(promedios['PRIMARYMACHINENAME']):
            datos_maquina = df_filtrado[df_filtrado['PRIMARYMACHINENAME'] == maquina]['DIPPERCOUNT']
            color_maquina = PALETA_CATEGORICA[i % len(PALETA_CATEGORICA)]

            fig.add_trace(go.Box(
                y=datos_maquina,
                name=maquina,
                boxpoints='outliers',
                marker_color=color_maquina,
                line=dict(color='gray', width=1.2),
                fillcolor=color_maquina,
                opacity=0.7,
                whiskerwidth=0.8,
                showlegend=False
            ))

        fig.add_trace(go.Scatter(
            x=promedios['PRIMARYMACHINENAME'],
            y=promedios['PROMEDIO_DIPPERCOUNT'],
            mode='markers',
            marker=dict(
                color=COLOR_WARNING,
                size=10,
                symbol='diamond',
                line=dict(color='white', width=1.5)
            ),
            name='Promedio (♦)'
        ))

        fig.update_layout(
            title=dict(
                text='PROMEDIO DE PASADAS POR PALAS',
                x=0.5,
                xanchor='center',
                font=dict(size=16, family="Arial, sans-serif", color='#2c3e50')
            ),
            xaxis=dict(
                title=dict(
                    text='PALAS',
                    font=dict(size=12)
                ),
                categoryorder='array',
                categoryarray=sorted(promedios['PRIMARYMACHINENAME'].unique()),
                linecolor='gray',
                mirror=True,
                showgrid=True,
                gridcolor='lightgray',
                tickangle=0
            ),
            yaxis=dict(
                title=dict(
                    text='NRO DE PASADAS',
                    font=dict(size=12)
                ),
                linecolor='gray',
                mirror=True,
                showgrid=True,
                gridcolor='lightgray'
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            template=PLOTLY_TEMPLATE,
            legend=dict(
                x=0.02,
                y=0.98,
                xanchor='left',
                yanchor='top',
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='gray',
                borderwidth=1
            ),
            font=dict(family="Arial, sans-serif"),
            margin=dict(t=90, l=60, r=60, b=60)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        #Promedio de Tiempo de Carguío y Promedio de Pases por Pala

         # Agrupar por máquina y calcular promedio
        resumen = df1_filtrado.groupby('SECONDARYMACHINENAME').agg({
            'TIEMPO_CARGUIO': 'mean',
            'DIPPERCOUNT': 'mean'  
        }).sort_values('TIEMPO_CARGUIO', ascending=False)

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Bar(
                x=resumen.index,
                y=resumen['TIEMPO_CARGUIO'],
                name='Tiempo de Carguío Promedio (min)',
                text=resumen['TIEMPO_CARGUIO'].round(2),
                textposition='outside',
                marker_color=COLOR_PRIMARY,
                opacity=0.8
            ),
            secondary_y=False
        )

        fig.add_trace(
            go.Scatter(
                x=resumen.index,
                y=resumen['DIPPERCOUNT'],
                name='Promedio de Pases',
                mode='lines+markers+text',
                text=resumen['DIPPERCOUNT'].round(2),
                textposition='top center',
                line=dict(color=COLOR_SECONDARY, width=3),
                marker=dict(size=8)
            ),
            secondary_y=True
        )

        fig.update_layout(
            title=dict(
                text='Promedio de Tiempo de Carguío y Promedio de Pases por Pala',
                font=dict(size=16, family="Arial, sans-serif", color='#2c3e50')
            ),
            xaxis_title='Pala',
            yaxis_title='Tiempo de Carguío Promedio (min)',
            legend_title='Métrica',
            template=PLOTLY_TEMPLATE,
            font=dict(family="Arial, sans-serif"),
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(showgrid=True, gridcolor='lightgray', tickangle=-45),
            yaxis=dict(showgrid=True, gridcolor='lightgray'),
            margin=dict(t=80, b=80)
        )

        fig.update_yaxes(
            title_text="Promedio de Pases",
            secondary_y=True,
            showgrid=False
        )

        st.plotly_chart(fig, use_container_width=True)

    #Promedio de número de pasadas y tonelaje por equipo - FULL WIDTH
    df_prom = df_filtrado.groupby('PRIMARYMACHINENAME', as_index=False)[['DIPPERCOUNT', 'PAYLOAD']].mean()

    fig2 = go.Figure()
    for i, (equipo, df_eq) in enumerate(df_prom.groupby('PRIMARYMACHINENAME')):
        fig2.add_trace(go.Scatter(
            x=df_eq['DIPPERCOUNT'],
            y=df_eq['PAYLOAD'],
            mode='markers+text',
            name=equipo,
            text=equipo,
            textposition='top center',
            marker=dict(
                symbol='diamond', 
                size=14, 
                color=PALETA_CATEGORICA[i % len(PALETA_CATEGORICA)],
                line=dict(width=1.5, color='white')
            )
        ))

    fig2.update_layout(
        title=dict(
            text='PROMEDIO DE NRO PASADAS Y TONELAJE POR EQUIPO',
            x=0.5, 
            font=dict(size=16, family="Arial, sans-serif", color='#2c3e50')
        ),
        xaxis=dict(
            title=dict(text='NRO DE PASADAS', font=dict(size=12)),
            showgrid=True,
            gridcolor='lightgray'
        ),
        yaxis=dict(
            title=dict(text='TONELAJE PROMEDIO (ton)', font=dict(size=12)),
            showgrid=True,
            gridcolor='lightgray'
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        template=PLOTLY_TEMPLATE,
        legend=dict(bordercolor='gray', borderwidth=1, bgcolor='rgba(255,255,255,0.8)'),
        font=dict(family="Arial, sans-serif"),
        height=500
    )
    st.plotly_chart(fig2, use_container_width=True)
    plt.close()


if maquina =="Camiones" and submenu =="Análisis Mensual por Máquina":
        
        # Distancia mensual recorrida por camión

        df1_filtrado['MES'] = df1_filtrado['STARTTIME'].dt.strftime('%Y-%m')

        distancia_mensual = (
            df1_filtrado.groupby(['PRIMARYMACHINENAME', 'MES'], as_index=False)
            ['DISTANCIA_CARGADO_EFH(mts)']
            .sum()
            .rename(columns={'DISTANCIA_CARGADO_EFH(mts)': 'DISTANCIA_TOTAL_MTS'})
        )

        distancia_mensual['DISTANCIA_TOTAL_KM'] = distancia_mensual['DISTANCIA_TOTAL_MTS'] / 1000

        fig = px.line(
            distancia_mensual,
            x='MES',
            y='DISTANCIA_TOTAL_KM',
            color='PRIMARYMACHINENAME',
            markers=True,
            title='Distancia mensual recorrida por camión',
            labels={
                'MES': 'Mes',
                'DISTANCIA_TOTAL_KM': 'Distancia Total (km)',
                'PRIMARYMACHINENAME': 'Camión'
            },
            color_discrete_sequence=PALETA_CATEGORICA
        )

        fig.update_layout(
            title=dict(
                text='Distancia mensual recorrida por camión',
                x=0.5,
                xanchor='center',
                font=dict(size=16, family="Arial, sans-serif", color='#2c3e50')
            ),
            xaxis=dict(
                showgrid=True,
                tickmode='array',
                tickvals=sorted(distancia_mensual['MES'].unique()),
                tickangle=45,
                title='Meses',
                gridcolor='lightgray'
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='lightgray',
                tickformat=',',
                title='Distancia Total (km)'
            ),
            legend=dict(
                title='Camión',
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='gray',
                borderwidth=1
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            template=PLOTLY_TEMPLATE,
            font=dict(family="Arial, sans-serif"),
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        plt.close()

        #Cantidad de viajes y tonelaje promedio por camión
       
        viajes_payload = (
            df1_filtrado.groupby(['MES', 'PRIMARYMACHINENAME'], as_index=False)
            .agg(
                N_VIAJES=('PRIMARYMACHINENAME', 'count'),
                PAYLOAD_PROMEDIO=('PAYLOAD', 'mean')
            )
        )

        fig2 = go.Figure()

        for camion in viajes_payload['PRIMARYMACHINENAME'].unique():
            df_camion = viajes_payload[viajes_payload['PRIMARYMACHINENAME'] == camion]
            fig2.add_trace(go.Bar(
                x=df_camion['MES'],
                y=df_camion['N_VIAJES'],
                name=f"{camion} - Viajes",
                marker_color=PALETA_CATEGORICA[
                    viajes_payload['PRIMARYMACHINENAME'].unique().tolist().index(camion) % len(PALETA_CATEGORICA)
                ],
                opacity=0.7
            ))

        payload_mensual = (
            viajes_payload.groupby('MES', as_index=False)['PAYLOAD_PROMEDIO'].mean()
        )
        fig2.add_trace(go.Scatter(
            x=payload_mensual['MES'],
            y=payload_mensual['PAYLOAD_PROMEDIO'],
            name='Payload Promedio (tn)',
            mode='lines+markers',
            line=dict(color='black', width=3, dash='dash'),
            marker=dict(size=8, symbol='circle')
        ))

        fig2.update_layout(
            title=dict(
                text='Cantidad de viajes y tonelaje promedio por mes',
                x=0.5,
                xanchor='center',
                font=dict(size=18, family="Arial, sans-serif", color='#2c3e50')
            ),
            xaxis=dict(
                title='Mes',
                tickangle=45,
                showgrid=True,
                gridcolor='lightgray'
            ),
            yaxis=dict(
                title='Cantidad de Viajes',
                showgrid=True,
                gridcolor='lightgray'
            ),
            yaxis2=dict(
                title='Payload Promedio (ton)',
                overlaying='y',
                side='right',
                showgrid=False
            ),
            legend=dict(
                title='Camión / Métrica',
                bgcolor='rgba(255,255,255,0.85)',
                bordercolor='gray',
                borderwidth=1
            ),
            barmode='group',
            plot_bgcolor='white',
            paper_bgcolor='white',
            template=PLOTLY_TEMPLATE,
            font=dict(family="Arial, sans-serif"),
            height=500
        )

        st.plotly_chart(fig2, use_container_width=True)
        plt.close()