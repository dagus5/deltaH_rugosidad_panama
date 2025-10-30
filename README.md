
# Δh – Índice de rugosidad (Norma FM Panamá) – SRTM (offline-friendly)

App **Streamlit** que calcula el **índice de rugosidad del terreno (Δh)** conforme a la **Norma FM Panamá**,
usando elevación **SRTM** (sin APIs externas, sin límites de tasa).

## Características
- Muestreo cada **500 m** entre **10–50 km** por radial.
- **Un azimut** o **varios** (lista personalizada).
- **Persistencia de resultados** en la sesión (historial).
- Mapas **Folium** y exportación **CSV/Excel**.
- Elevación con **srtm.py** (descarga/caché automática de tiles HGT).

## Instalación
```bash
pip install -r requirements.txt
```

## Ejecutar
```bash
streamlit run app.py
```

## Notas normativas
- Δh = h10 − h90 (percentiles 10 % y 90 %) en el trayecto **10–50 km**, cada **500 m**.
- ΔF = `1.9 - 0.03*Δh*(1 + f/300)` (f en MHz).
