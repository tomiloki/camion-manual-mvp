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

# --- 2. FUNCIONES DE GOOGLE API (v0.21 - CACHÉ 100% MANUAL) ---
def obtener_geocoding(direccion):
    
    if direccion in st.session_state.geocoding_cache:
        return st.session_state.geocoding_cache[direccion]
    
    geocode_url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = { 'address': f"{direccion}, Chile", 'key': API_KEY }
    try:
        with st.spinner(f"Geocodificando (API): {direccion}..."):
            response = requests.get(geocode_url, params=params)
            results = response.json().get('results', [])
            if results:
                location = results[0]['geometry']['location']
                lat_lon = f"{location['lat']},{location['lng']}"
                st.session_state.geocoding_cache[direccion] = lat_lon
                return lat_lon
            else:
                st.warning(f"No se pudo geocodificar: {direccion}.")
                return None
    except Exception as e:
        st.error(f"Error en API Geocoding: {e}")
        return None

def obtener_matriz_distancia_chunked(_direcciones_tuple, max_elements=100, max_retries=3):
    direcciones = list(_direcciones_tuple)
    n = len(direcciones)
    dist_matrix_km = np.zeros((n, n))
    time_matrix_min = np.zeros((n, n))
    chunk_size = int(np.sqrt(max_elements))
    
    progress_bar = st.progress(0.0, text="Calculando Matriz de Distancias... (0%)")
    total_chunks = int(np.ceil(n / chunk_size)) ** 2
    completed_chunks = 0

    with st.spinner(f"Calculando Matriz de Distancias ({n}x{n} puntos)... (Llamada a API)"):
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
    st.success("Matriz de Distancias calculada (API).")
    return np.round(dist_matrix_km, 1), np.round(time_matrix_min, 1)

# --- 3. FUNCIONES DE OR-TOOLS (Lógica del Colab - Pasos 5, 6, 8) ---
def crear_modelo_datos(matriz_distancias, num_camiones):
    data = {}
    data['distance_matrix'] = matriz_distancias.tolist()
    data['num_vehicles'] = num_camiones
    data['depot'] = 0
    return data

# --- ¡FUNCIÓN MODIFICADA v0.30! ---
# Ahora devuelve una lista de métricas por camión
def imprimir_solucion_streamlit(manager, routing, solution, data, paradas_texto, matriz_tiempos):
    st.subheader("Resultados de Optimización:")
    
    total_distance = 0
    total_time = 0
    max_route_time = 0
    camiones_usados = 0
    total_paradas_asignadas = 0
    
    lista_metricas_camiones = [] # <-- ¡NUEVO!

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

        if parada_count > 0:
            camiones_usados += 1
            total_paradas_asignadas += parada_count
            paradas_ruta_texto += "Bodega"
            
            total_distance += route_distance
            total_time += route_time
            if route_time > max_route_time:
                max_route_time = route_time

            # --- ¡NUEVO v0.30! Guardamos las métricas de este camión ---
            metricas_camion = {
                "Camión": f"🚚 Camión {vehicle_id + 1}",
                "Paradas": parada_count,
                "Distancia (km)": route_distance,
                "Tiempo (min)": route_time
            }
            lista_metricas_camiones.append(metricas_camion)

            # Tarjeta de v0.29 (sin cambios)
            with st.container(border=True):
                st.markdown(f'**🚚 Camión {vehicle_id + 1}: {parada_count} paradas | {route_distance:.1f} km | {route_time/60.0:.1f} hrs**')
                st.caption(f"Detalle ruta: {paradas_ruta_texto}")
                st.markdown("---")
                st.markdown("**Links de Google Maps:**")
                
                max_paradas_link = 8 
                num_links = int(np.ceil(len(route_indices) / max_paradas_link))
                current_origin_node = 0 

                for i in range(num_links):
                    start_idx = i * max_paradas_link
                    end_idx = (i + 1) * max_paradas_link
                    chunk_indices = route_indices[start_idx : end_idx]
                    
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
                        destination_text = paradas_texto[0] 
                        waypoint_texts = [paradas_texto[idx] for idx in chunk_indices] 
                        paradas_link_final = [origin_text] + waypoint_texts + [destination_text]
                    
                    base_url = "https://www.google.com/maps/dir/"
                    encoded_paradas = [requests.utils.quote(p) for p in paradas_link_final]
                    link = base_url + "/".join(encoded_paradas)
                    
                    st.markdown(f"**Link {i+1}/{num_links}:**")
                    st.link_button(f"Abrir Ruta {i+1}/{num_links} en Google Maps ↗️", link, use_container_width=True)
                    st.code(link)
            
            st.markdown(" ") 
        
    st.subheader(f'**Distancia Total: {total_distance:.1f} km**')
    
    # Devolvemos la lista de métricas
    return total_distance, total_time, camiones_usados, max_route_time, total_paradas_asignadas, lista_metricas_camiones

# --- 4. INTERFAZ DE USUARIO ---

st.title("CamiON - Mago de Oz (v0.30) 🚚") # <-- TÍTULO ACTUALIZADO
st.write("Tu co-piloto para optimizar costos operativos ✨")

st.header("1. Ingresar Datos de la Operación")

# --- ¡NUEVO v0.30! Callback para limpiar texto ---
def limpiar_texto():
    st.session_state.direcciones_texto = ""
# --- FIN DEL CAMBIO ---

col1, col2 = st.columns(2)
with col1:
    DIRECCION_BODEGA = st.text_input("Dirección Bodega (Inicio/Fin)", "Av. Americo Vespucio 1925, Conchalí, Santiago")
    NUMERO_DE_CAMIONES = st.number_input("¿Cuántos camiones?", min_value=1, max_value=20, value=1)
    COSTO_KM_CLP = st.number_input(
        "Costo Operativo por KM (CLP) (Opcional)", # <-- (Opcional) movido aquí
        min_value=0, 
        max_value=2000, 
        value=0,
        help="Define el costo variable por kilómetro. Considera: combustible, peajes, desgaste de neumáticos y mantención proporcional."
    )
    
with col2:
    # --- ¡NUEVO v0.30! Se añade la 'key' ---
    texto_paradas = st.text_area("Pega aquí las direcciones (UNA por línea)", 
                                 height=250,
                                 placeholder="Pega tu lista de 10 direcciones aquí...",
                                 key="direcciones_texto")
    # --- FIN DEL CAMBIO ---

# --- ¡NUEVO v0.30! Botones en columnas ---
col_opt, col_limpiar = st.columns([3, 1]) # Botón de optimizar es 3 veces más grande
with col_opt:
    boton_optimizar = st.button("✨ OPTIMIZAR RUTA", type="primary", use_container_width=True)
with col_limpiar:
    st.button("🧹 Limpiar Lista", on_click=limpiar_texto, use_container_width=True)
# --- FIN DEL CAMBIO ---


# --- 5. LÓGICA DE EJECUCIÓN (v0.30) ---

if 'matrix_cache' not in st.session_state:
    st.session_state.matrix_cache = {}
if 'geocoding_cache' not in st.session_state:
    st.session_state.geocoding_cache = {}

direcciones_validas_texto = []
direcciones_para_api_latlon = []

if boton_optimizar:
    if not texto_paradas or not DIRECCION_BODEGA:
        st.error("Por favor, ingresa la dirección de la bodega y al menos una parada.")
    else:
        st.header("2. Procesando...")
        try:
            
            # --- PASO A: Geocodificación (Lógica v0.21 - Robusta y Manual) ---
            st.subheader("Paso A: Geocodificando direcciones...")
            
            lista_paradas_input = [linea.strip() for linea in texto_paradas.split('\n') if linea.strip()]
            
            puntos_validos_temporal = []

            bodega_latlon = obtener_geocoding(DIRECCION_BODEGA) 
            if bodega_latlon:
                puntos_validos_temporal.append( (DIRECCION_BODEGA, bodega_latlon) )
            else:
                st.error("Error geocodificando la Bodega. La app no puede continuar.")
                st.stop()
            
            for direccion_texto in lista_paradas_input:
                parada_latlon = obtener_geocoding(direccion_texto) 
                if parada_latlon:
                    puntos_validos_temporal.append( (direccion_texto, parada_latlon) )
            
            direcciones_validas_texto = [item[0] for item in puntos_validos_temporal]
            direcciones_para_api_latlon = [item[1] for item in puntos_validos_temporal]

            st.info(f"Geocodificación completa. {len(direcciones_validas_texto)} puntos totales (1 Bodega + {len(direcciones_validas_texto)-1} paradas).")
            
            if len(direcciones_validas_texto) < 2:
                st.error("Se necesita al menos 1 parada válida para optimizar.")
            else:
                
                # --- PASO B: Matriz de Distancia (con Caché Manual v0.16) ---
                st.subheader("Paso B: Calculando matriz de distancias...")
                
                cache_key = tuple(direcciones_para_api_latlon)
                
                if cache_key in st.session_state.matrix_cache:
                    st.success("✅ Matriz de Distancias obtenida de la caché.")
                    matriz_km, matriz_min = st.session_state.matrix_cache[cache_key]
                
                else:
                    st.warning("⚠️ No se encontró la matriz en caché. Calculando con API...")
                    matriz_km, matriz_min = obtener_matriz_distancia_chunked(cache_key)
                    
                    st.session_state.matrix_cache[cache_key] = (matriz_km, matriz_min)
                
                if np.all(matriz_km <= 0) and len(direcciones_validas_texto) > 1:
                    st.error("Error en el cálculo de la Matriz. Todas las distancias son 0 o -1. Revisa los permisos de la API 'Distance Matrix'.")
                    st.stop()
                
                # --- PASO C: OR-Tools (v0.9 - CON BALANCEO DE TIEMPO) ---
                st.subheader("Paso C: Optimizando con OR-Tools (Balanceando carga)...")
                
                data = crear_modelo_datos(matriz_km, NUMERO_DE_CAMIONES)
                manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                                    data['num_vehicles'], data['depot'])
                routing = pywrapcp.RoutingModel(manager)

                def distance_callback(from_index, to_index):
                    from_node = manager.IndexToNode(from_index)
                    to_node = manager.IndexToNode(to_index)
                    dist = data['distance_matrix'][from_node][to_node]
                    return 9999999 if dist <= 0 else int(dist * 10)
                
                transit_callback_index = routing.RegisterTransitCallback(distance_callback)
                routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
                
                def time_callback(from_index, to_index):
                    from_node = manager.IndexToNode(from_index)
                    to_node = manager.IndexToNode(to_index)
                    tiempo = matriz_min[from_node][to_node]
                    return 9999999 if tiempo < 0 else int(tiempo)

                time_callback_index = routing.RegisterTransitCallback(time_callback)

                routing.AddDimension(
                    time_callback_index,
                    0,     # slack
                    3000,  # Capacidad máxima de tiempo (un número grande)
                    True,  # Empezar acumulado en cero
                    'Time'
                )
                time_dimension = routing.GetDimensionOrDie('Time')

                search_parameters = pywrapcp.DefaultRoutingSearchParameters()
                time_dimension.SetGlobalSpanCostCoefficient(100) # Penalización ALTA
                
                solution = routing.SolveWithParameters(search_parameters)

                # --- ¡PASO D MODIFICADO v0.30! ---
                if solution:
                    st.header("3. ¡Rutas Optimizadas!")
                    # 1. Imprimir las rutas y OBTENER las métricas (incluyendo la lista de desglose)
                    dist_optimizada, tpo_total_hh, camiones_usados, tpo_jornada, paradas_asignadas, lista_metricas_camiones = imprimir_solucion_streamlit(
                        manager, routing, solution, data, 
                        direcciones_validas_texto, matriz_min 
                    )
                    
                    # 3. Tabla de Resumen Total (la de v0.29)
                    st.subheader("Resumen Total de la Operación")
                    resumen_data = {
                        "Métrica": [
                            "🚚 Camiones Usados",
                            "📍 Total Paradas Asignadas",
                            "🏁 Distancia Total",
                            "⏰ Tiempo Total (Horas-Hombre)",
                            "☀️ Jornada (Camión más ocupado)"
                        ],
                        "Resultado": [
                            f"{camiones_usados} de {NUMERO_DE_CAMIONES} (disponibles)",
                            f"{paradas_asignadas} (de {len(direcciones_validas_texto)-1} paradas)",
                            f"{dist_optimizada:.1f} km",
                            f"{tpo_total_hh/60.0:.1f} Horas",
                            f"**{tpo_jornada/60.0:.1f} Horas**"
                        ]
                    }
                    
                    if COSTO_KM_CLP > 0:
                        costo_operativo_total = dist_optimizada * COSTO_KM_CLP
                        resumen_data["Métrica"].insert(3, "💰 Costo Operativo Total")
                        resumen_data["Resultado"].insert(3, f"${costo_operativo_total:,.0f} CLP")

                    
                    df_resumen = pd.DataFrame(resumen_data).set_index("Métrica")
                    st.dataframe(df_resumen, width='stretch')
                    st.balloons()
                    
                    # --- ¡NUEVO v0.30! Tabla de Desglose por Camión ---
                    st.subheader("Desglose por Camión")
                    
                    # 1. Crear el dataframe a partir de la lista
                    df_desglose = pd.DataFrame(lista_metricas_camiones)
                    
                    # 2. Formatear columnas
                    df_desglose["Tiempo (hrs)"] = (df_desglose["Tiempo (min)"] / 60.0).round(1)
                    df_desglose["Distancia (km)"] = df_desglose["Distancia (km)"].round(1)
                    
                    # 3. Añadir costos si aplica
                    columnas_ordenadas = ["Paradas", "Distancia (km)", "Tiempo (hrs)"]
                    if COSTO_KM_CLP > 0:
                        df_desglose["Costo Operativo (CLP)"] = (df_desglose["Distancia (km)"] * COSTO_KM_CLP)
                        columnas_ordenadas.append("Costo Operativo (CLP)")

                    # 4. Añadir Fila de Total
                    total_row = pd.DataFrame({
                        "Camión": "**Total**",
                        "Paradas": paradas_asignadas,
                        "Distancia (km)": dist_optimizada,
                        "Tiempo (hrs)": tpo_total_hh / 60.0,
                        "Costo Operativo (CLP)": (dist_optimizada * COSTO_KM_CLP) if COSTO_KM_CLP > 0 else 0
                    }, index=[0])
                    
                    df_desglose = pd.concat([df_desglose, total_row], ignore_index=True)
                    
                    # 5. Formatear y mostrar
                    df_desglose = df_desglose.set_index("Camión")
                    df_desglose = df_desglose.drop(columns=["Tiempo (min)"]) # Eliminar la columna de minutos
                    
                    # Formateo final de números
                    if COSTO_KM_CLP > 0:
                         df_desglose["Costo Operativo (CLP)"] = df_desglose["Costo Operativo (CLP)"].apply(lambda x: f"${x:,.0f}")
                    
                    st.dataframe(df_desglose[columnas_ordenadas], width='stretch')
                    # --- FIN DEL NUEVO PASO ---

                else:
                    st.error("No se encontró una solución. Revisa las direcciones o el número de camiones.")

        except Exception as e:
            st.error(f"¡ERROR CRÍTICO DURANTE LA EJECUCIÓN!")
            st.exception(e)