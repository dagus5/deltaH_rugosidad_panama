
# Δh – Índice de rugosidad (Norma FM Panamá)

Aplicación **Streamlit** que calcula el **índice de rugosidad del terreno (Δh)** conforme a la **Norma Técnica de Radiodifusión Analógica en FM (Panamá)**:
- Muestra perfiles de terreno cada **500 m** entre **10 y 50 km** desde la antena transmisora.
- Permite **un azimut** o **múltiples** (ingresados por el usuario).
- Calcula **Δh = h10 - h90** y la corrección **ΔF = 1.9 - 0.03·Δh·(1 + f/300)**.
- Visualiza los radiales en un **mapa interactivo**.
- Exporta resultados a **CSV** y **Excel**.

## Requisitos
- Python 3.9+
- Conexión a Internet (usa la API gratuita de **Open‑Meteo Elevation**).

## Instalación
```bash
pip install -r requirements.txt
```

## Ejecución
```bash
streamlit run app.py
```

## Uso
1. Introduce **latitud** y **longitud** de la antena.
2. Selecciona **un azimut** o escribe una lista separada por comas.
3. Indica la **frecuencia (MHz)** para calcular **ΔF**.
4. Pulsa **Calcular Δh** para ver tabla, mapa y descargas.

## Notas normativas
- Muestreo: **cada 500 m** entre **10–50 km** desde el transmisor.
- **Δh**: diferencia entre alturas **rebasadas** en el **10 %** y **90 %** del trayecto.
- **ΔF**: `1.9 - 0.03*Δh*(1 + f/300)` (f en MHz).
