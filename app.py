
import streamlit as st
import pandas as pd
import numpy as np
import math
import requests
from math import radians, sin, cos, atan2, sqrt
from io import BytesIO
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Δh – Índice de rugosidad (Norma FM Panamá)", layout="wide")

# --------- Helpers: geodesic direct (spherical Earth approximate) ---------
R_EARTH_M = 6371000.0

def destination_point(lat_deg, lon_deg, bearing_deg, distance_m):
    lat1 = math.radians(lat_deg)
    lon1 = math.radians(lon_deg)
    brng = math.radians(bearing_deg)
    dr = distance_m / R_EARTH_M
    lat2 = math.asin(math.sin(lat1) * math.cos(dr) + math.cos(lat1) * math.sin(dr) * math.cos(brng))
    lon2 = lon1 + math.atan2(math.sin(brng) * math.sin(dr) * math.cos(lat1),
                             math.cos(dr) - math.sin(lat1) * math.sin(lat2))
    return math.degrees(lat2), (math.degrees(lon2) + 540) % 360 - 180  # normalize lon

# --------- Elevation via Open-Meteo API (free, no token) ---------
def fetch_elevations(lat_list, lon_list, batch_size=500):
    """Return list of elevations (m) using Open-Meteo Elevation API. Preserves order."""
    elevations = []
    for i in range(0, len(lat_list), batch_size):
        lats = lat_list[i:i+batch_size]
        lons = lon_list[i:i+batch_size]
        url = "https://api.open-meteo.com/v1/elevation"
        params = {"latitude": ",".join(map(lambda x: f"{x:.6f}", lats)),
                  "longitude": ",".join(map(lambda x: f"{x:.6f}", lons))}
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        elevations.extend(data.get("elevation", [None]*len(lats)))
    return elevations

def compute_delta_h(elevations_m):
    """Per norma: Δh = h10 - h90, percentiles over heights between 10-50 km at 500 m spacing."""
    arr = np.array([e for e in elevations_m if e is not None], dtype=float)
    if arr.size == 0:
        return None, None, None
    h10 = float(np.percentile(arr, 10))
    h90 = float(np.percentile(arr, 90))
    delta_h = h10 - h90
    return delta_h, h10, h90

def deltaF_from_deltaH(delta_h, freq_mhz):
    # Norma: ΔF = 1.9 - 0.03 * Δh * (1 + f/300)
    return 1.9 - 0.03 * (delta_h) * (1.0 + freq_mhz/300.0)

st.title("🗺️ Índice de rugosidad Δh (Norma FM – Panamá)")
st.caption("Muestreo cada 500 m entre 10–50 km desde la antena, por radial/es seleccionados. Elevación: Open‑Meteo.")

with st.sidebar:
    st.header("Parámetros")
    lat = st.number_input("Latitud de la antena (°)", value=8.806600, format="%.6f")
    lon = st.number_input("Longitud de la antena (°)", value=-82.540300, format="%.6f")
    freq = st.number_input("Frecuencia (MHz) para ΔF", value=100.0, min_value=50.0, max_value=300.0, step=0.1, format="%.1f")
    st.markdown("---")
    mode = st.radio("Modo de azimuts", ["Un solo azimut", "Varios azimuts"], index=1)
    if mode == "Un solo azimut":
        az_list = [st.number_input("Azimut (°)", value=0.0, min_value=0.0, max_value=359.9, step=0.1)]
    else:
        az_str = st.text_input("Lista de azimuts (°) separados por coma", value="0,45,90,135,180,225,270,315")
        try:
            az_list = [float(x.strip()) for x in az_str.split(",") if x.strip()!=""]
        except Exception:
            az_list = []
            st.error("Formato de lista inválido. Usa valores numéricos separados por coma.")
    st.markdown("---")
    step_m = 500
    st.text("Muestreo: cada 500 m (norma)")
    start_km = 10.0
    end_km = 50.0

run = st.button("Calcular Δh")

if run and len(az_list) == 0:
    st.error("Debes especificar al menos un azimut.")
    st.stop()

if run:
    results = []
    map_center = (lat, lon)
    fmap = folium.Map(location=map_center, zoom_start=8, control_scale=True)

    for az in az_list:
        distances_m = list(range(int(start_km*1000), int(end_km*1000)+1, step_m))
        lats, lons = [], []
        for d in distances_m:
            plat, plon = destination_point(lat, lon, az, d)
            lats.append(plat); lons.append(plon)

        try:
            elev = fetch_elevations(lats, lons)
        except Exception as e:
            st.error(f"Error obteniendo elevaciones para azimut {az}°: {e}")
            continue

        delta_h, h10, h90 = compute_delta_h(elev)
        if delta_h is None:
            st.warning(f"Sin datos de elevación para azimut {az}°")
            continue

        dF = deltaF_from_deltaH(delta_h, freq)
        row = {
            "Azimut (°)": az,
            "Δh (m)": round(delta_h, 2),
            "h10 (m)": round(h10, 2),
            "h90 (m)": round(h90, 2),
            "ΔF (dB)": round(dF, 2),
            "Puntos": len(elev),
        }
        results.append(row)
        pts = list(zip(lats, lons))
        folium.PolyLine(pts, weight=3, opacity=0.7).add_to(fmap)

    if len(results)==0:
        st.stop()

    res_df = pd.DataFrame(results).sort_values("Azimut (°)").reset_index(drop=True)
    st.subheader("Resultados por azimut")
    st.dataframe(res_df, use_container_width=True)

    st.markdown("**Resumen (promedios):**")
    st.write({
        "Δh promedio (m)": round(res_df["Δh (m)"].mean(), 2),
        "ΔF promedio (dB)": round(res_df["ΔF (dB)"].mean(), 2),
    })

    folium.Marker(location=map_center, tooltip="Transmisor").add_to(fmap)
    st_folium(fmap, width=None, height=500)

    def to_excel_bytes(df):
        from openpyxl import Workbook
        from openpyxl.utils.dataframe import dataframe_to_rows
        wb = Workbook()
        ws = wb.active
        ws.title = "DeltaH"
        for r in dataframe_to_rows(df, index=False, header=True):
            ws.append(r)
        ws["G1"] = "Δh según Norma FM Panamá: Δh = h10 - h90 (10–50 km, cada 500 m)"
        ws["G2"] = "Corrección ΔF = 1.9 - 0.03*Δh*(1 + f/300)"
        out = BytesIO()
        wb.save(out)
        return out.getvalue()

    csv_bytes = res_df.to_csv(index=False).encode("utf-8")
    xlsx_bytes = to_excel_bytes(res_df)

    st.download_button("⬇️ Descargar CSV", data=csv_bytes, file_name="deltaH_resultados.csv", mime="text/csv")
    st.download_button("⬇️ Descargar Excel", data=xlsx_bytes, file_name="deltaH_resultados.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.caption("Cálculo conforme a la Norma Técnica de Radiodifusión Analógica en FM (Panamá): Δh = h10 - h90 entre 10–50 km; ΔF = 1.9 - 0.03Δh(1+f/300).")
