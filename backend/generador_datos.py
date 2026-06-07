import requests
import pandas as pd
import time
import urllib.parse
import re
import os

# =========================================================
# CARGA DE DATOS ORIGINAL
# =========================================================
sesiones_icmla = {
    "Special Session 1: Deep Learning and Applications": [
        "A Comparative Analysis of Transformer and LSTM Models for Detecting Suicidal Ideation on Reddit",
        "Mental Stress Classification by Attention-based CNN-LSTM Algorithm of Electrocardiogram Signal",
        "Multi-Sphere Anomaly Detection with Hyperspherical Layers",
        "Depression Classification Algorithm Based on Voice Signals Using MFCC and CNN Autoencoders",
        "WeedVision: Multi-Stage Growth and Classification of Weeds using DETR and RetinaNet for Precision Agriculture",
        "Efficient Retraining for Continuous Operability through Strategic Directives",
        "Evaluating Adversarial Attacks on Traffic Sign Classifiers beyond Standard Baselines",
        "Real-Time Automatic Checkout via Prompt-Based Product Extraction and Cross-Domain Learning",
        "Edge-Centric Real-Time Segmentation for Autonomous Underwater Cave Exploration",
        "Turn Down The Noise: Perceptually Constrained Attacks for Multi-Label Audio Classification",
        "Finding an Optimal Small Sample of Training Dataset for Computer Vision Deep-Learning Models",
        "TriplePlay: Enhancing Federated Learning with CLIP for Non-IID Data and Resource Efficiency"
    ],
    "Special Session 2: Machine Learning for Natural Language Processing": [
        "Tell Me More! Using Multiple Features for Binary Text Classification with a Zero-Shot Model",
        "Sentiment Classification on Twitter(X) Through Ensemble Deep Random Vector Functional Links",
        "Systematical randomness assignment for the level of manipulation in text augmentation",
        "Using LLMs to Establish Implicit User Sentiment of Software Desirability",
        "Detecting Cyberbullying in Visual Content: A Large Vision-Language Model Approach",
        "Domain-Specific Retrieval-Augmented Generation Using Vector Stores, Knowledge Graphs, and Tensor Factorization",
        "Empathetic Reflective Response Generation: Towards Conversation Models for Online Mental Health Support",
        "LLM-based Sign Language Production"
    ],
    "Special Session 3: Machine Learning for Predictive Models in Engineering Applications": [
        "Towards Highly Efficient Anomaly Detection for Predictive Maintenance",
        "An Evaluation and Comparison of Machine Learning Methods for Prediction of Lubricant Film Thickness",
        "On the Effectiveness of Heterogeneous Ensemble Methods for Re-identification",
        "High-speed Deformation Prediction in Selective Laser Melting Using Context-adaptive Neural Networks",
        "AssemAl: Interpretable Image-Based Anomaly Detection for Manufacturing Pipelines",
        "Brazilian free-energy market mid- and long-term forecasting using multi-source ensemble solution",
        "Fusion of Real and Synthetic Subtracted Contrast-Enhanced Mammograms for Enhanced Tumor Detection",
        "A Multi-layered Expert Recommender System for Enhanced Customer Support",
        "Harnessing Machine Learning and Stock Market Techniques for Signal Detection in Underwater Sensing Technologies",
        "Advancing Energy Monitoring: Deep Learning for Automated Non-Smart Gas Meter Readings",
        "Aircraft Engine Remaining Useful Life (RUL) Prediction Using Machine Learning",
        "Hierarchical Supervised Monte Carlo Ensemble Learning",
        "Enhancing Pipeline Monitoring: Optimizing Window Size with Monte Carlo Search and CB-AttentionNet",
        "A Game-Theoretic Analytical Framework for Approximation with Soft Sets"
    ],
    "Special Session 4: Quantum Machine Learning Algorithms and Applications": [
        "Hierarchical Learning for Training Large-Scale Variational Quantum Circuits",
        "Mitigating the Effects of Concept Drift from Data Streams in Quantum Machine Learning",
        "Utilizing a Hybrid Matrix Product State and Variational Quantum Circuit Architecture for the Detection of Kidney Diseases"
    ],
    "Special Session 5: Machine Learning for Earth Observation (ML4EO)": [
        "Detecting Important Drivers of Gridded Population Modeling With Machine Learning",
        "Predicting Particulate Matter Values in Metropolitan Areas Using Machine Learning",
        "Characterizing the Impact of Common Electro-Optical Sensor Anomalies on Maritime Image Classifiers"
    ]
}

ANIO = 2024

def obtener_abstract_y_keywords(titulo):
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    parametros = {
        "query": titulo,
        "limit": 1,
        "fields": "title,abstract,fieldsOfStudy" 
    }
    try:
        respuesta = requests.get(url, params=parametros)
        if respuesta.status_code == 200:
            datos = respuesta.json()
            if datos['total'] > 0:
                paper = datos['data'][0]
                campos = paper.get('fieldsOfStudy', [])
                campos = [c for c in campos if c is not None]
                keywords = ", ".join(campos) if campos else ""
                
                return {
                    "Title": paper.get('title', titulo),
                    "Keywords": keywords,
                    "Abstract": paper.get('abstract', '') or 'No abstract'
                }
    except Exception:
        pass
    return {"Title": titulo, "Keywords": "", "Abstract": "No encontrado"}

def recuperar_datos_openalex(titulo):
    """
    Versión mejorada: Intenta rescatar tanto el abstract como los keywords (conceptos).
    """
    url = f"https://api.openalex.org/works?filter=display_name.search:{urllib.parse.quote(titulo)}"
    abstract = None
    keywords = None
    try:
        respuesta = requests.get(url)
        if respuesta.status_code == 200:
            datos = respuesta.json()
            if datos.get('results'):
                resultado = datos['results'][0]
                
                # 1. Rescatar Abstract
                inverted = resultado.get('abstract_inverted_index')
                if inverted:
                    max_pos = max([pos for positions in inverted.values() for pos in positions])
                    abstract_arr = [""] * (max_pos + 1)
                    for word, positions in inverted.items():
                        for pos in positions:
                            abstract_arr[pos] = word
                    abstract = " ".join(abstract_arr)
                
                # 2. Rescatar Keywords (OpenAlex los llama 'concepts')
                conceptos = resultado.get('concepts', [])
                if conceptos:
                    # Extraemos los nombres de los conceptos con mayor relevancia
                    keywords = ", ".join([c['display_name'] for c in conceptos if 'display_name' in c][:5])
                    
    except:
        pass
    
    return abstract, keywords

def recuperar_abstract_crossref(titulo):
    url = f"https://api.crossref.org/works?query.title={urllib.parse.quote(titulo)}&select=abstract&rows=1"
    try:
        respuesta = requests.get(url)
        if respuesta.status_code == 200:
            datos = respuesta.json()
            items = datos['message']['items']
            if items and 'abstract' in items[0]:
                abstract = items[0]['abstract']
                abstract = re.sub(r'<[^>]+>', '', abstract)
                return abstract
    except:
        pass
    return None

def generar_keywords_desde_titulo(titulo):
    # SISTEMA DE EMERGENCIA: Extraer palabras clave directamente del título
    palabras = re.findall(r'\b[A-Za-z]{5,}\b', str(titulo))
    return ", ".join(set(palabras))

def limpiar_texto_estructural(texto):
    if pd.isna(texto):
        return ""
    texto = str(texto)
    texto = re.sub(r'<.*?>', '', texto)
    texto = re.sub(r'[\n\t\r]', ' ', texto)
    texto = re.sub(r'[^\x00-\x7F]+', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

if __name__ == "__main__":
    print("Iniciando extracción masiva para todas las sesiones...\n")
    resultados = []
    paper_id_actual = 1

    for nombre_sesion, titulos in sesiones_icmla.items():
        print(f"--- Procesando: {nombre_sesion} ---")
        for idx, titulo in enumerate(titulos):
            print(f"[{paper_id_actual}] Buscando: {titulo}")
            datos_api = obtener_abstract_y_keywords(titulo)
            fila = {
                "Paper_Id": paper_id_actual,
                "Title": datos_api["Title"],
                "Keywords": datos_api["Keywords"],
                "Abstract": datos_api["Abstract"],
                "Session": nombre_sesion,
                "Year": ANIO
            }
            resultados.append(fila)
            paper_id_actual += 1
            time.sleep(1.5)
        print()

    df = pd.DataFrame(resultados)
    columnas_mendeley = ["Paper_Id", "Title", "Keywords", "Abstract", "Session", "Year"]
    df = df[columnas_mendeley]
    
    # --- NUEVA LÓGICA DE RUTAS A PRUEBA DE BALAS ---
    # 1. Obtener la ruta exacta de donde está este script (carpeta backend)
    directorio_script = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Subir un nivel y apuntar a la carpeta 'dataset'
    carpeta_dataset = os.path.join(directorio_script, "..", "dataset")
    
    # 3. Forzar la creación de la carpeta si por alguna razón no existe
    os.makedirs(carpeta_dataset, exist_ok=True)
    
    # 4. Crear la ruta final para el primer archivo
    ruta_dataset_incompleto = os.path.join(carpeta_dataset, "Celine_ICMLA_2024_Dataset_Completo.csv")
    # -----------------------------------------------

    df.to_csv(ruta_dataset_incompleto, index=False, encoding="utf-8")
    print(f"¡Extracción exitosa! Guardado en {ruta_dataset_incompleto}.")

    print("\nIniciando rescate de abstracts y keywords faltantes...")
    df = pd.read_csv(ruta_dataset_incompleto)
    
    for index, row in df.iterrows():
        falta_abstract = pd.isna(row['Abstract']) or row['Abstract'] in ["No encontrado", "No abstract", ""]
        falta_keywords = pd.isna(row['Keywords']) or row['Keywords'] == ""

        if falta_abstract or falta_keywords:
            titulo = row['Title']
            print(f"Buscando info faltante para: {titulo[:60]}...")
            
            # Llamamos a nuestra nueva súper función
            abstract_rescatado, keywords_rescatadas = recuperar_datos_openalex(titulo)
            
            # Si faltaba el abstract y lo encontramos, lo guardamos
            if falta_abstract:
                if abstract_rescatado:
                    df.at[index, 'Abstract'] = abstract_rescatado
                    print("  -> ¡Abstract recuperado con OpenAlex!")
                else:
                    # Si OpenAlex falló, intentamos con Crossref como plan B
                    abstract_cross = recuperar_abstract_crossref(titulo)
                    if abstract_cross:
                        df.at[index, 'Abstract'] = abstract_cross
                        print("  -> ¡Abstract recuperado con Crossref!")
            
            # MODIFICACIÓN APLICADA: Si faltaban los keywords, intentamos API o auto-generación
            if falta_keywords:
                if keywords_rescatadas:
                    df.at[index, 'Keywords'] = keywords_rescatadas
                    print(f"  -> ¡Keywords recuperados! ({keywords_rescatadas})")
                else:
                    kw_emergencia = generar_keywords_desde_titulo(titulo)
                    df.at[index, 'Keywords'] = kw_emergencia
                    print(f"  -> Keywords auto-generadas desde el título: ({kw_emergencia})")
                
            time.sleep(1) # Pausa amigable para no saturar las APIs

    print("\nIniciando inyección de datos faltantes y limpieza...")
    abstracts_rescate = {
        "Domain-Specific Retrieval-Augmented Generation Using Vector Stores, Knowledge Graphs, and Tensor Factorization": "Large Language Models (LLMs) are pre-trained on large-scale corpora and excel in numerous general natural language processing (NLP) tasks. Additionally, fine tuning LLMs' intrinsic knowledge to highly specific domains is an expensive and time consuming process. In this paper, we introduce SMART-SLIC, a highly domain-specific LLM framework, that integrates RAG with Knowledge Graphs (KG) and a vector store (VS) that store factual domain specific information. Pairing our RAG with a domain-specific KG and VS enables the development of chatbots that attribute the source of information, mitigate hallucinations, lessen the need for fine-tuning, and excel in highly domain-specific tasks.",
        "AssemAl: Interpretable Image-Based Anomaly Detection for Manufacturing Pipelines": "Anomaly detection in manufacturing pipelines remains a critical challenge, intensified by the complexity and variability of industrial environments. This paper introduces AssemAI, an interpretable image-based anomaly detection system tailored for smart manufacturing pipelines. The authors created an image dataset specifically focused on the rocket assembly pipeline environment. The research fine-tunes the YOLO-FF object detection model and introduces a tailored anomaly detection model for assembly pipelines. The framework incorporates domain knowledge to enhance model accuracy and interpretability, utilizing SCORE-CAM and ontology that contextualizes the model's outputs.",
        "A Game-Theoretic Analytical Framework for Approximation with Soft Sets": "Addressing uncertainty issues is a significant challenge in decision-making. Soft set theory is designed to assist in complex decision-making scenarios where multiple approximations are involved. We propose a game-theoretic framework to address these challenges effectively, offering desirable outcomes in scenarios with inherent conflicts when using soft sets. Experimental results indicate that the model not only can achieve a balance among various parameters or conflicting decision goals, but also improves approximation performance across accuracy, precision, recall, and F1-score."
    }

    for titulo_faltante, abstract_real in abstracts_rescate.items():
        df.loc[df['Title'] == titulo_faltante, 'Abstract'] = abstract_real

    df = df[~df['Abstract'].isin(["No encontrado", "No abstract"])]
    df = df.dropna(subset=['Title', 'Abstract'])

    df['Title'] = df['Title'].apply(limpiar_texto_estructural)
    df['Abstract'] = df['Abstract'].apply(limpiar_texto_estructural)
    df['Keywords'] = df['Keywords'].apply(limpiar_texto_estructural)

    # Crear la ruta final para el segundo archivo (Limpio)
    ruta_dataset_limpio = os.path.join(carpeta_dataset, "Celine_ICMLA_2024_Limpio.csv")
    df.to_csv(ruta_dataset_limpio, index=False, encoding="utf-8")

    print(f"-> Limpieza finalizada. Total de documentos: {len(df)}")
    print(f"¡Fase 1 completada! Archivo definitivo guardado en:\n{ruta_dataset_limpio}")