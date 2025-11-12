# --- IMPORTS ---
import streamlit as st
import pandas as pd
import numpy as np
import requests 
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import json 
import time

# --- 1. CONFIGURACIÓN INICIAL (v0.6) ---
st.set_page_config(page_title="CamiON Mago de Oz", layout="wide")

try:
    API_KEY = st.secrets["GOOGLE_API_KEY"]
except KeyError:
    st.error("⛔ ERROR: API Key no encontrada. Revisa tu .streamlit/secrets.toml")
    st.stop()

# --- 2. FUNCIONES DE GOOGLE API (Lógica del Colab + Caché v0.6) ---
# (Sin cambios)
@st.cache_data 
def obtener_geocoding(direccion):
    geocode_url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = { 'address': f"{direccion}, Chile", 'key': API_KEY }
    try:
        with st.spinner(f"Geocodificando: {direccion}..."):
            response = requests.get(geocode_url, params=params)
            results = response.json().get('results', [])
            if results:
                location = results[0]['geometry']['location']
                return f"{location['lat']},{location['lng']}"
            else:
                st.warning(f"No se pudo geocodificar: {direccion}.")
                return None
    except Exception as e:
        st.error(f"Error en API Geocoding: {e}")
        return None

@st.cache_data 
def obtener_matriz_distancia_chunked(_direcciones_tuple, max_elements=100, max_retries=3):
    direcciones = list(_direcciones_tuple)
    n = len(direcciones)
    dist_matrix_km = np.zeros((n, n))
    time_matrix_min = np.zeros((n, n))
    chunk_size = int(np.sqrt(max_elements))
    
    progress_bar = st.progress(0.0, text="Calculando Matriz de Distancias... (0%)")
    total_chunks = int(np.ceil(n / chunk_size)) ** 2
    completed_chunks = 0

    with st.spinner(f"Calculando Matriz de Distancias ({n}x{n} puntos)..."):
        for i_chunk in range(0, n, chunk_size):
            for j_chunk in range(0, n, chunk_size):
                i_start, i_end = i_chunk, min(i_chunk + chunk_size, n)
                j_start, j_end = j_chunk, min(j_chunk + chunk_size, n)
                origins = "|".join(direcciones[i_start:i_end])
                destinations = "|".join(direcciones[j_start:j_end])
                api_url = "https://maps.googleapis.com/maps/api/distancematrix/json"
                params = { 'origins': origins, 'destinations': destinations, 'key': API_KEY, 'units': 'metric', 'mode': 'driving' }

                for attempt in range(max_retries):
                    try:
                        response = requests.get(api_url, params=params)
                        data = response.json()
                        
                        if data['status'] != 'OK':
                            print(f"--- ERROR DE GOOGLE API (Matriz): {data.get('status')} ---")
                            print(data)
                            if data['status'] == 'REQUEST_DENIED':
                                st.error(f"Error de API: {data.get('error_message', 'REQUEST_DENIED')}. Revisa tus permisos en Google Cloud.")
                                st.stop()

                        if data['status'] == 'OK':
                            for i, row in enumerate(data['rows']):
                                for j, element in enumerate(row['elements']):
                                    if element['status'] == 'OK':
                                        dist_matrix_km[i + i_start, j + j_start] = element['distance']['value'] / 1000.0
                                        time_matrix_min[i + i_start, j + j_start] = element['duration']['value'] / 60.0
                                    else:
                                        dist_matrix_km[i + i_start, j + j_start] = -1
                                        time_matrix_min[i + i_start, j + j_start] = -1
                            break 
                        elif data['status'] == 'OVER_QUERY_LIMIT':
                            time.sleep(2**attempt)
                        else:
                            break 
                    except KeyError:
                        print("--- ERROR DE KEYERROR (Respuesta inesperada) ---")
                        print(data)
                        st.error(f"Respuesta inesperada de Google. Revisa la consola.")
                        st.json(data)
                        break
                    except Exception as e:
                        st.error(f"Excepción en API (Matriz): {e}")
                        time.sleep(1)
                
                completed_chunks += 1
                progress_bar.progress(completed_chunks / total_chunks, text=f"Calculando Matriz... ({int(100*completed_chunks/total_chunks)}%)")

    progress_bar.empty()
    st.success("Matriz de Distancias calculada.")
    return np.round(dist_matrix_km, 1), np.round(time_matrix_min, 1)

# --- 3. FUNCIONES DE OR-TOOLS (Lógica del Colab - Pasos 5, 6, 8) ---
# (Sin cambios)
def crear_modelo_datos(matriz_distancias, num_camiones):
    data = {}
    data['distance_matrix'] = matriz_distancias.tolist()
    data['num_vehicles'] = num_camiones
    data['depot'] = 0
    return data

def imprimir_solucion_streamlit(manager, routing, solution, data, paradas_texto, matriz_tiempos):
    total_distance = 0
    total_time = 0
    st.subheader("Resultados de Optimización:")

    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        route_distance = 0
        route_time = 0
        route_indices = []
        paradas_ruta_texto = ""
        parada_count = 0 

        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            
            if node_index != 0: 
                parada_count += 1
                paradas_ruta_texto += f"Parada {node_index} -> "
                route_indices.append(node_index)
            else:
                paradas_ruta_texto += "Bodega -> "

            previous_index = index
            index = solution.Value(routing.NextVar(index))
            
            dist_segmento = routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
            route_distance += (dist_segmento / 10.0)
            
            prev_node = manager.IndexToNode(previous_index)
            next_node = manager.IndexToNode(index)
            tiempo_segmento = matriz_tiempos[prev_node][next_node]
            route_time += 0 if tiempo_segmento < 0 else tiempo_segmento

        if parada_count == 0:
            continue

        paradas_ruta_texto += "Bodega"
        st.markdown(f'**Ruta para Camión {vehicle_id + 1}:** ({parada_count} paradas)')
        st.caption(paradas_ruta_texto)
        st.markdown(f'* Distancia: **{route_distance:.1f} km**')
        st.markdown(f'* Tiempo estimado: **{route_time:.0f} min** ({route_time/60.0:.1f} hrs)')
        
        st.markdown("Links de Google Maps (divididos para ser funcionales):")
        
        max_paradas_link = 8 
        num_links = int(np.ceil(len(route_indices) / max_paradas_link))
            
        current_origin_node = 0 # El primer origen es la Bodega (nodo 0)

        for i in range(num_links):
            start_idx = i * max_paradas_link
            end_idx = (i + 1) * max_paradas_link
            chunk_indices = route_indices[start_idx : end_idx]
            
            # Bug v0.12: ¿Qué pasa si chunk_indices está vacío?
            # No debería pasar si parada_count > 0, pero por si acaso:
            if not chunk_indices:
                continue

            origin_text = paradas_texto[current_origin_node]
            destination_node = chunk_indices[-1]
            destination_text = paradas_texto[destination_node]
            waypoint_indices = chunk_indices[:-1]
            waypoint_texts = [paradas_texto[idx] for idx in waypoint_indices]
            
            if not waypoint_texts:
                paradas_link_final = [origin_text, destination_text]
            else:
                paradas_link_final = [origin_text] + waypoint_texts + [destination_text]
            
            current_origin_node = destination_node

            if i == num_links - 1:
                destination_text = paradas_texto[0] # Destino final es Bodega
                waypoint_texts = [paradas_texto[idx] for idx in chunk_indices] 
                paradas_link_final = [origin_text] + waypoint_texts + [destination_text]
            
            base_url = "https://www.google.com/maps/dir/"
            encoded_paradas = [requests.utils.quote(p) for p in paradas_link_final]
            link = base_url + "/".join(encoded_paradas)
            
            st.markdown(f"**Link {i+1}/{num_links}:**") 
            st.code(link) 

        st.markdown("---")
        total_distance += route_distance
        total_time += route_time
        
    st.subheader(f'**Distancia Total (Todos los camiones): {total_distance:.1f} km**')
    return total_distance, total_time

def calcular_ahorro_baseline(matriz_dist_km, matriz_tiempos_min, costo_km_clp):
    try:
        dist_baseline = 0
        tiempo_baseline = 0
        n_puntos = len(matriz_dist_km)
        for i in range(1, n_puntos):
            dist_ida = matriz_dist_km[0][i]
            dist_vuelta = matriz_dist_km[i][0]
            if dist_ida > 0 and dist_vuelta > 0:
                dist_baseline += dist_ida + dist_vuelta
                tiempo_baseline += matriz_tiempos_min[0][i] + matriz_tiempos_min[i][0]
        costo_baseline = dist_baseline * costo_km_clp
        return dist_baseline, tiempo_baseline, costo_baseline
    except Exception as e:
        st.warning(f"No se pudo calcular el línea base (manual): {e}")
        return 0, 0, 0

# --- 4. INTERFAZ DE USUARIO ---

st.title("CamiON - Mago de Oz (v0.13 - Bugfix) 🚚") # <-- TÍTULO ACTUALIZADO
st.write("Herramienta interna para optimización manual de rutas.")
# (Resto de la UI sin cambios)
st.header("1. Ingresar Datos de la Operación")
col1, col2 = st.columns(2)
with col1:
    DIRECCION_BODEGA = st.text_input("Dirección Bodega (Inicio/Fin)", "Av. Americo Vespucio 1925, Conchalí, Santiago")
    NUMERO_DE_CAMIONES = st.number_input("¿Cuántos camiones?", min_value=1, max_value=20, value=2) 
    
with col2:
    texto_paradas = st.text_area("Pega aquí las direcciones (UNA por línea)", 
                                 height=250,
                                 placeholder="Pega tu lista de 21 direcciones aquí...")

st.subheader("Datos para Análisis de Ahorro")
COSTO_KM_CLP = st.number_input("Costo Operativo por KM (CLP)", min_value=100, max_value=1000, value=286)

boton_optimizar = st.button("✨ OPTIMIZAR RUTA", type="primary", width='stretch')


# --- 5. LÓGICA DE EJECUCIÓN (v0.9 - CON BALANCEO DE CARGA) ---

if boton_optimizar:
    if not texto_paradas or not DIRECCION_BODEGA:
        st.error("Por favor, ingresa la dirección de la bodega y al menos una parada.")
    else:
        st.header("2. Procesando...")
        try:
            
            # --- ¡NUEVO! PASO A: Geocodificación (v0.13 - A prueba de balas) ---
            st.subheader("Paso A: Geocodificando direcciones...")
            
            lista_paradas_input = [linea.strip() for linea in texto_paradas.split('\n') if linea.strip()]
            
            # 1. Crear UNA lista temporal para garantizar el sync
            puntos_validos_temporal = []

            # 2. Geocodificar Bodega
            bodega_latlon = obtener_geocoding(DIRECCION_BODEGA)
            if bodega_latlon:
                puntos_validos_temporal.append( (DIRECCION_BODEGA, bodega_latlon) )
            else:
                st.error("Error geocodificando la Bodega. La app no puede continuar.")
                st.stop()
            
            # 3. Geocodificar Paradas
            for direccion_texto in lista_paradas_input:
                parada_latlon = obtener_geocoding(direccion_texto)
                
                # Solo si la geocodificación fue exitosa, la agregamos
                if parada_latlon:
                    puntos_validos_temporal.append( (direccion_texto, parada_latlon) )
            
            # 4. "Descomprimir" la lista temporal en las dos listas finales
            # Esto GARANTIZA que tienen el mismo largo y el mismo orden
            direcciones_validas_texto = [item[0] for item in puntos_validos_temporal]
            direcciones_para_api_latlon = [item[1] for item in puntos_validos_temporal]

            # --- FIN DEL NUEVO PASO A ---

            st.info(f"Geocodificación completa. {len(direcciones_validas_texto)} puntos totales (1 Bodega + {len(direcciones_validas_texto)-1} paradas).")
            
            if len(direcciones_validas_texto) < 2:
                st.error("Se necesita al menos 1 parada válida para optimizar.")
            else:
                # --- PASO B: Matriz de Distancia ---
                st.subheader("Paso B: Calculando matriz de distancias...")
                matriz_km, matriz_min = obtener_matriz_distancia_chunked(tuple(direcciones_para_api_latlon))
                
                if np.all(matriz_km <= 0) and len(direcciones_validas_texto) > 1:
                    st.error("Error en el cálculo de la Matriz. Todas las distancias son 0 o -1. Revisa los permisos de la API 'Distance Matrix'.")
                    st.stop()
                
                # --- PASO C: OR-Tools (v0.9 - CON BALANCEO DE TIEMPO) ---
                st.subheader("Paso C: Optimizando con OR-Tools (Balanceando carga)...")
                
                data = crear_modelo_datos(matriz_km, NUMERO_DE_CAMIONES)
                manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                                    data['num_vehicles'], data['depot'])
                routing = pywrapcp.RoutingModel(manager)

                # 1. Callback de Distancia
                def distance_callback(from_index, to_index):
                    from_node = manager.IndexToNode(from_index)
                    to_node = manager.IndexToNode(to_index)
                    dist = data['distance_matrix'][from_node][to_node]
                    return 9999999 if dist <= 0 else int(dist * 10)
                
                transit_callback_index = routing.RegisterTransitCallback(distance_callback)
                routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
                
                # 2. Callback de TIEMPO (para la dimensión de balanceo)
                def time_callback(from_index, to_index):
                    from_node = manager.IndexToNode(from_index)
                    to_node = manager.IndexToNode(to_index)
                    tiempo = matriz_min[from_node][to_node]
                    return 9999999 if tiempo < 0 else int(tiempo)

                time_callback_index = routing.RegisterTransitCallback(time_callback)

                # 3. Añadir Dimensión de Tiempo
                routing.AddDimension(
                    time_callback_index,
                    0,     # slack
                    3000,  # Capacidad máxima de tiempo (un número grande)
                    True,  # Empezar acumulado en cero
                    'Time'
                )
                time_dimension = routing.GetDimensionOrDie('Time')

                # 4. ¡LA CLAVE! Poner el "costo" en el MINMAX (Global Span)
                search_parameters = pywrapcp.DefaultRoutingSearchParameters()
                time_dimension.SetGlobalSpanCostCoefficient(100) # Penalización ALTA
                
                solution = routing.SolveWithParameters(search_parameters)

                # --- PASO D: Resultados y Ahorro ---
                if solution:
                    st.header("3. ¡Rutas Optimizadas!")
                    dist_optimizada, tpo_optimizado = imprimir_solucion_streamlit(
                        manager, routing, solution, data, 
                        direcciones_validas_texto, matriz_min 
                    )
                    costo_optimizado = dist_optimizada * COSTO_KM_CLP
                    
                    st.header("4. Análisis de Ahorro")
                    dist_base, tpo_base, costo_base = calcular_ahorro_baseline(matriz_km, matriz_min, COSTO_KM_CLP)
                    
                    if costo_base > 0:
                        ahorro_dist = dist_base - dist_optimizada
                        ahorro_costo = costo_base - costo_optimizado
                        
                        f_costo_base = f"${costo_base:,.0f}"
                        f_costo_opt = f"${costo_optimizado:,.0f}"
                        f_ahorro_costo = f"${ahorro_costo:,.0f}"
                        
                        reporte_data = {
                            "MÉTRICA": ["Distancia Total", "Costo Operativo"],
                            "BASELINE (Manual)": [f"{dist_base:.1f} km", f"{f_costo_base} CLP"],
                            "CAMION (Optimizado)": [f"{dist_optimizada:.1f} km", f"{f_costo_opt} CLP"]
                        }
                        df_reporte = pd.DataFrame(reporte_data).set_index("MÉTRICA")
                        st.dataframe(df_reporte, width='stretch')
                        
                        st.markdown("---")
                        st.success(f"**AHORRO TOTAL: {ahorro_dist:.1f} km ({f_ahorro_costo} CLP)**")
                        st.balloons()
                    else:
                        st.warning("No se pudo calcular el ahorro (costo base fue 0).")
                else:
                    st.error("No se encontró una solución. Revisa las direcciones o el número de camiones.")

        except Exception as e:
            st.error(f"¡ERROR CRÍTICO DURANTE LA EJECUCIÓN!")
            st.exception(e)