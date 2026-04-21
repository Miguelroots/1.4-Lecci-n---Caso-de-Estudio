#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CLUSTERING DE PRODUCTOS EN PYTHON + PANDAS + SCIKIT-LEARN
Segmentación de Inventario en Logística
Caso de Estudio: Análisis del Ciclo de Vida de la Ciencia de Datos
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
import warnings
import os

warnings.filterwarnings('ignore')

# Configurar estilo de visualizaciones
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

print("\n" + "="*80)
print("CLUSTERING DE PRODUCTOS EN PYTHON")
print("Segmentación de Inventario en Logística")
print("="*80 + "\n")

# ============================================================================
# FASE 1: CARGA DE DATOS
# ============================================================================
print("[FASE 1] Cargando datos...")

try:
    # Cargar datos maestros de productos
    df_productos = pd.read_csv(r'producto_tabla\producto_tabla.csv')
    print(f"✓ Tabla de productos cargada: {df_productos.shape[0]} productos")
    
    # Cargar transacciones
    df_transacciones = pd.read_csv(r'test\test.csv')
    print(f"✓ Tabla de transacciones cargada: {df_transacciones.shape[0]:,} registros")
except Exception as e:
    print(f"✗ Error cargando datos: {e}")
    exit(1)

# ============================================================================
# FASE 2: EXPLORACIÓN INICIAL
# ============================================================================
print("\n[FASE 2] Explorando datos...")
print(f"Columnas productos: {df_productos.columns.tolist()}")
print(f"Columnas transacciones: {df_transacciones.columns.tolist()}")
print(f"Rango de fechas: {df_transacciones.iloc[:, 0].min()} a {df_transacciones.iloc[:, 0].max()}")

# ============================================================================
# FASE 3: PREPROCESAMIENTO
# ============================================================================
print("\n[FASE 3] Preprocesamiento de datos...")

# Eliminar nulos
df_transacciones_clean = df_transacciones.dropna()
print(f"✓ Registros después de eliminar nulos: {df_transacciones_clean.shape[0]:,}")

# Deduplicación
df_trans_clean = df_transacciones_clean.drop_duplicates()
print(f"✓ Registros únicos: {df_trans_clean.shape[0]:,}")

# ============================================================================
# FASE 4: FEATURE ENGINEERING
# ============================================================================
print("\n[FASE 4] Ingeniería de características...")

# Obtener nombre de columnas (flexible para nombres)
cols = df_trans_clean.columns.tolist()
id_col = cols[0]  # Primera columna: ID transacción
fecha_col = cols[1]  # Segunda: Fecha
producto_col = cols[2]  # Tercera: Producto
cliente_col = cols[3]  # Cuarta: Cliente
cantidad_col = cols[4] if len(cols) > 4 else None  # Quinta: Cantidad (si existe)

# Crear características
feature_cols = ['Volumen_Total', 'Frecuencia_Pedidos', 'Concentracion_Ventas']

df_features = df_trans_clean.groupby(producto_col).agg({
    id_col: 'count',  # Volumen total
    producto_col: 'nunique',  # Productos únicos (será 1, pero lo mantenemos)
    cliente_col: 'nunique'  # Clientes únicos
}).reset_index()

df_features.columns = [producto_col, 'Volumen_Total', 'Dummy', 'Clientes_Unicos']
df_features = df_features.drop('Dummy', axis=1)

# Calcular concentración de ventas
df_features['Concentracion_Ventas'] = df_features['Volumen_Total'] / (df_features['Clientes_Unicos'] + 1)

# Seleccionar features para clustering
df_clustering = df_features[[producto_col, 'Volumen_Total', 'Frecuencia_Pedidos', 'Concentracion_Ventas']].copy()
df_clustering.columns = [producto_col, 'Volumen_Total', 'Frecuencia_Pedidos', 'Concentracion_Ventas']

print(f"✓ Features creados para {df_clustering.shape[0]} productos")
print(f"  - Volumen Total (min/max): {df_clustering['Volumen_Total'].min():.0f} / {df_clustering['Volumen_Total'].max():.0f}")
print(f"  - Frecuencia de Pedidos (min/max): {df_clustering['Frecuencia_Pedidos'].min():.0f} / {df_clustering['Frecuencia_Pedidos'].max():.0f}")
print(f"  - Concentración (min/max): {df_clustering['Concentracion_Ventas'].min():.2f} / {df_clustering['Concentracion_Ventas'].max():.2f}")

# ============================================================================
# FASE 5: EXPLORACIÓN DE DATOS (EDA)
# ============================================================================
print("\n[FASE 5] Análisis Exploratorio de Datos...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Distribución de Features de Clustering', fontsize=16, fontweight='bold')

axes[0, 0].hist(df_clustering['Volumen_Total'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
axes[0, 0].set_title('Volumen Total de Transacciones', fontweight='bold')
axes[0, 0].set_xlabel('Volumen')
axes[0, 0].set_ylabel('Frecuencia')

axes[0, 1].hist(df_clustering['Frecuencia_Pedidos'], bins=50, color='coral', edgecolor='black', alpha=0.7)
axes[0, 1].set_title('Frecuencia de Pedidos', fontweight='bold')
axes[0, 1].set_xlabel('Frecuencia')
axes[0, 1].set_ylabel('Frecuencia')

axes[1, 0].hist(df_clustering['Concentracion_Ventas'], bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
axes[1, 0].set_title('Concentración de Ventas', fontweight='bold')
axes[1, 0].set_xlabel('Concentración')
axes[1, 0].set_ylabel('Frecuencia')

# Correlación
corr_data = df_clustering[['Volumen_Total', 'Frecuencia_Pedidos', 'Concentracion_Ventas']].corr()
sns.heatmap(corr_data, annot=True, fmt='.3f', cmap='coolwarm', ax=axes[1, 1], cbar_kws={'label': 'Correlación'})
axes[1, 1].set_title('Matriz de Correlación', fontweight='bold')

plt.tight_layout()
plt.savefig('01_Distribuciones.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico guardado: 01_Distribuciones.png")
plt.close()

# ============================================================================
# FASE 6: NORMALIZACIÓN
# ============================================================================
print("\n[FASE 6] Normalización de datos...")

X = df_clustering[['Volumen_Total', 'Frecuencia_Pedidos', 'Concentracion_Ventas']].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"✓ Datos normalizados con StandardScaler")
print(f"  Media (después de normalizar): {X_scaled.mean(axis=0)}")
print(f"  Desv. Est (después de normalizar): {X_scaled.std(axis=0)}")

# ============================================================================
# FASE 7: DETERMINACIÓN DE K ÓPTIMO (K-MEANS)
# ============================================================================
print("\n[FASE 7] Determinando K óptimo para K-Means...")

inertias = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
    print(f"  K={k}: Inércia={kmeans.inertia_:.2f}, Silhouette={silhouette_scores[-1]:.4f}")

# Encontrar K óptimo
best_k_silhouette = k_range[np.argmax(silhouette_scores)]
print(f"✓ K óptimo (Silhouette Score): K={best_k_silhouette}")

# ============================================================================
# FASE 8: ENTRENAMIENTO K-MEANS FINAL
# ============================================================================
print(f"\n[FASE 8] Entrenando K-Means con K={best_k_silhouette}...")

kmeans_final = KMeans(n_clusters=best_k_silhouette, random_state=42, n_init=10)
df_clustering['Cluster_KMeans'] = kmeans_final.fit_predict(X_scaled)
silhouette_kmeans = silhouette_score(X_scaled, df_clustering['Cluster_KMeans'])

print(f"✓ K-Means entrenado")
print(f"  Silhouette Score: {silhouette_kmeans:.4f}")
print(f"  Distribución de clusters: {df_clustering['Cluster_KMeans'].value_counts().sort_index().to_dict()}")

# ============================================================================
# FASE 9: DETERMINACIÓN DE COMPONENTES ÓPTIMOS (GMM)
# ============================================================================
print("\n[FASE 9] Determinando componentes óptimos para GMM...")

bic_scores = []
silhouette_gmm_scores = []

for n_comp in k_range:
    gmm = GaussianMixture(n_components=n_comp, random_state=42, n_init=10)
    gmm.fit(X_scaled)
    bic_scores.append(gmm.bic(X_scaled))
    labels = gmm.predict(X_scaled)
    silhouette_gmm_scores.append(silhouette_score(X_scaled, labels))
    print(f"  Componentes={n_comp}: BIC={bic_scores[-1]:.2f}, Silhouette={silhouette_gmm_scores[-1]:.4f}")

# Encontrar óptimo
best_gmm_components = k_range[np.argmin(bic_scores)]
print(f"✓ Componentes óptimos (BIC mínimo): {best_gmm_components}")

# ============================================================================
# FASE 10: ENTRENAMIENTO GMM FINAL
# ============================================================================
print(f"\n[FASE 10] Entrenando GMM con {best_gmm_components} componentes...")

gmm_final = GaussianMixture(n_components=best_gmm_components, random_state=42, n_init=10)
df_clustering['Cluster_GMM'] = gmm_final.fit_predict(X_scaled)
silhouette_gmm = silhouette_score(X_scaled, df_clustering['Cluster_GMM'])

print(f"✓ GMM entrenado")
print(f"  Silhouette Score: {silhouette_gmm:.4f}")
print(f"  Distribución de clusters: {df_clustering['Cluster_GMM'].value_counts().sort_index().to_dict()}")

# ============================================================================
# FASE 11: VISUALIZACIÓN - K ÓPTIMO Y GMM
# ============================================================================
print("\n[FASE 11] Generando gráficos de optimización...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# K óptimo
axes[0].plot(k_range, inertias, 'o-', color='steelblue', linewidth=2, markersize=8, label='Inércia')
axes[0].axvline(x=best_k_silhouette, color='red', linestyle='--', linewidth=2, label=f'K óptimo={best_k_silhouette}')
axes[0].set_xlabel('K (Número de Clusters)', fontweight='bold')
axes[0].set_ylabel('Inércia', fontweight='bold')
axes[0].set_title('K-Means: Método del Codo', fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# GMM BIC
axes[1].plot(k_range, bic_scores, 'o-', color='coral', linewidth=2, markersize=8, label='BIC')
axes[1].axvline(x=best_gmm_components, color='red', linestyle='--', linewidth=2, label=f'Óptimo={best_gmm_components}')
axes[1].set_xlabel('Número de Componentes', fontweight='bold')
axes[1].set_ylabel('BIC Score', fontweight='bold')
axes[1].set_title('GMM: Criterio BIC', fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].legend()

plt.tight_layout()
plt.savefig('02_Optimizacion_Modelos.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico guardado: 02_Optimizacion_Modelos.png")
plt.close()

# ============================================================================
# FASE 12: CARACTERIZACIÓN DE CLUSTERS
# ============================================================================
print("\n[FASE 12] Caracterizando clusters K-Means...")

cluster_stats = []
for cluster_id in sorted(df_clustering['Cluster_KMeans'].unique()):
    cluster_data = df_clustering[df_clustering['Cluster_KMeans'] == cluster_id]
    
    vol_mean = cluster_data['Volumen_Total'].mean()
    freq_mean = cluster_data['Frecuencia_Pedidos'].mean()
    conc_mean = cluster_data['Concentracion_Ventas'].mean()
    size = len(cluster_data)
    
    # Lógica de nombrado
    vol_median = df_clustering['Volumen_Total'].median()
    freq_median = df_clustering['Frecuencia_Pedidos'].median()
    
    if vol_mean > vol_median and freq_mean > freq_median:
        if conc_mean > df_clustering['Concentracion_Ventas'].median():
            nombre = "Alta Rotación Centralizada"
        else:
            nombre = "Alta Rotación Dispersa"
    elif vol_mean < vol_median and freq_mean < freq_median:
        nombre = "Baja Rotación"
    else:
        nombre = "Rotación Media"
    
    cluster_stats.append({
        'Cluster': cluster_id,
        'Nombre': nombre,
        'Productos': size,
        'Volumen_Promedio': vol_mean,
        'Frecuencia_Promedio': freq_mean,
        'Concentracion_Promedio': conc_mean
    })
    
    print(f"  Cluster {cluster_id} ({nombre}):")
    print(f"    - Productos: {size}")
    print(f"    - Volumen promedio: {vol_mean:.0f}")
    print(f"    - Frecuencia promedio: {freq_mean:.1f}")
    print(f"    - Concentración promedio: {conc_mean:.2f}")

df_cluster_stats = pd.DataFrame(cluster_stats)

# ============================================================================
# FASE 13: VISUALIZACIÓN 2D
# ============================================================================
print("\n[FASE 13] Generando visualizaciones 2D...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# K-Means
scatter1 = axes[0].scatter(df_clustering['Volumen_Total'], 
                           df_clustering['Frecuencia_Pedidos'],
                           c=df_clustering['Cluster_KMeans'],
                           cmap='viridis',
                           s=100,
                           alpha=0.6,
                           edgecolors='black')
axes[0].set_xlabel('Volumen Total', fontweight='bold')
axes[0].set_ylabel('Frecuencia de Pedidos', fontweight='bold')
axes[0].set_title(f'K-Means Clustering (K={best_k_silhouette})', fontweight='bold')
axes[0].grid(True, alpha=0.3)
plt.colorbar(scatter1, ax=axes[0], label='Cluster')

# GMM
scatter2 = axes[1].scatter(df_clustering['Volumen_Total'], 
                           df_clustering['Frecuencia_Pedidos'],
                           c=df_clustering['Cluster_GMM'],
                           cmap='plasma',
                           s=100,
                           alpha=0.6,
                           edgecolors='black')
axes[1].set_xlabel('Volumen Total', fontweight='bold')
axes[1].set_ylabel('Frecuencia de Pedidos', fontweight='bold')
axes[1].set_title(f'GMM Clustering ({best_gmm_components} componentes)', fontweight='bold')
axes[1].grid(True, alpha=0.3)
plt.colorbar(scatter2, ax=axes[1], label='Cluster')

plt.tight_layout()
plt.savefig('03_Clusters_2D.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico guardado: 03_Clusters_2D.png")
plt.close()

# ============================================================================
# FASE 14: VISUALIZACIÓN 3D
# ============================================================================
print("\n[FASE 14] Generando visualizaciones 3D...")

fig = plt.figure(figsize=(16, 6))

# K-Means 3D
ax1 = fig.add_subplot(121, projection='3d')
scatter1 = ax1.scatter(df_clustering['Volumen_Total'],
                       df_clustering['Frecuencia_Pedidos'],
                       df_clustering['Concentracion_Ventas'],
                       c=df_clustering['Cluster_KMeans'],
                       cmap='viridis',
                       s=100,
                       alpha=0.6,
                       edgecolors='black')
ax1.set_xlabel('Volumen Total', fontweight='bold')
ax1.set_ylabel('Frecuencia de Pedidos', fontweight='bold')
ax1.set_zlabel('Concentración de Ventas', fontweight='bold')
ax1.set_title(f'K-Means 3D (K={best_k_silhouette})', fontweight='bold')

# GMM 3D
ax2 = fig.add_subplot(122, projection='3d')
scatter2 = ax2.scatter(df_clustering['Volumen_Total'],
                       df_clustering['Frecuencia_Pedidos'],
                       df_clustering['Concentracion_Ventas'],
                       c=df_clustering['Cluster_GMM'],
                       cmap='plasma',
                       s=100,
                       alpha=0.6,
                       edgecolors='black')
ax2.set_xlabel('Volumen Total', fontweight='bold')
ax2.set_ylabel('Frecuencia de Pedidos', fontweight='bold')
ax2.set_zlabel('Concentración de Ventas', fontweight='bold')
ax2.set_title(f'GMM 3D ({best_gmm_components} componentes)', fontweight='bold')

plt.tight_layout()
plt.savefig('04_Clusters_3D.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico guardado: 04_Clusters_3D.png")
plt.close()

# ============================================================================
# FASE 15: EXPORTAR RESULTADOS
# ============================================================================
print("\n[FASE 15] Exportando resultados...")

# Crear tabla de resultados con nombres descriptivos
df_resultados = df_clustering[[producto_col, 'Cluster_KMeans', 'Cluster_GMM',
                               'Volumen_Total', 'Frecuencia_Pedidos', 'Concentracion_Ventas']].copy()

# Agregar nombres de clusters K-Means
nombre_map = dict(zip(df_cluster_stats['Cluster'], df_cluster_stats['Nombre']))
df_resultados['Nombre_Cluster_KMeans'] = df_resultados['Cluster_KMeans'].map(nombre_map)

df_resultados.to_csv('resultados_clustering.csv', index=False, encoding='utf-8')
print(f"✓ Resultados exportados: resultados_clustering.csv ({len(df_resultados)} productos)")

# ============================================================================
# FASE 16: CONCLUSIONES Y RECOMENDACIONES
# ============================================================================
print("\n" + "="*80)
print("CONCLUSIONES Y RECOMENDACIONES DE INVENTARIO")
print("="*80)

print(f"\n📊 RESULTADOS DEL ANÁLISIS:")
print(f"   Total de productos analizados: {len(df_clustering)}")
print(f"   K-Means óptimo: {best_k_silhouette} clusters (Silhouette: {silhouette_kmeans:.4f})")
print(f"   GMM óptimo: {best_gmm_components} componentes (Silhouette: {silhouette_gmm:.4f})")

print(f"\n🎯 SEGMENTACIÓN K-MEANS:")
for _, row in df_cluster_stats.iterrows():
    print(f"\n   {row['Nombre'].upper()}")
    print(f"   Cluster ID: {int(row['Cluster'])}")
    print(f"   Productos: {int(row['Productos'])}")
    print(f"   Volumen promedio: {row['Volumen_Promedio']:.0f} transacciones")
    print(f"   Frecuencia promedio: {row['Frecuencia_Promedio']:.1f} pedidos")
    print(f"   Concentración: {row['Concentracion_Promedio']:.2f}")
    
    # Recomendaciones por cluster
    nombre = row['Nombre']
    if 'Alta Rotación Centralizada' in nombre:
        print(f"   💡 ESTRATEGIA: Alto stock, pocos proveedores, distribución concentrada")
    elif 'Alta Rotación Dispersa' in nombre:
        print(f"   💡 ESTRATEGIA: Stock alto, múltiples proveedores, distribución amplia")
    elif 'Baja Rotación' in nombre:
        print(f"   💡 ESTRATEGIA: Stock bajo, proveedores bajo demanda, monitoreo especial")
    else:
        print(f"   💡 ESTRATEGIA: Stock moderado, modelo de reabastecimiento flexible")

print("\n" + "="*80)
print("✅ ANÁLISIS COMPLETO")
print("="*80)
print(f"\n📁 Archivos generados:")
print(f"   • 01_Distribuciones.png - Histogramas y correlaciones")
print(f"   • 02_Optimizacion_Modelos.png - Elbow method y BIC")
print(f"   • 03_Clusters_2D.png - Visualización 2D de clusters")
print(f"   • 04_Clusters_3D.png - Visualización 3D de clusters")
print(f"   • resultados_clustering.csv - Tabla con asignación de clusters")
print("\n" + "="*80 + "\n")
