from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import os
import time

# Importamos tu clase desde el archivo que creamos antes
from motor_busqueda import MotorUPScholar

app = FastAPI(title="API UPScholar")

# =========================================================
# CONFIGURACIÓN DE RUTAS DINÁMICAS
# =========================================================
directorio_actual = os.path.dirname(os.path.abspath(__file__))
ruta_dataset = os.path.join(directorio_actual, "..", "dataset", "Celine_ICMLA_2024_Limpio.csv")
ruta_frontend = os.path.join(directorio_actual, "..", "frontend")

# =========================================================
# INICIALIZACIÓN DEL MOTOR (Se ejecuta 1 sola vez al arrancar)
# =========================================================
print("Inicializando Motor UPScholar... (Cargando datos y modelos LLM)")
# Si el archivo CSV aún no existe, el servidor fallará, 
# asegúrate de haber corrido generador_datos.py primero.
if os.path.exists(ruta_dataset):
    motor = MotorUPScholar(ruta_dataset)
    print("¡Motor listo y cargado en memoria!")
else:
    print("ADVERTENCIA: No se encontró el dataset. Ejecuta generador_datos.py primero.")

# =========================================================
# CONFIGURACIÓN DEL FRONTEND
# =========================================================
templates = Jinja2Templates(directory=ruta_frontend)
# Carpeta para archivos estáticos (CSS, Imágenes)
ruta_static = os.path.join(ruta_frontend, "static")
os.makedirs(ruta_static, exist_ok=True)
app.mount("/static", StaticFiles(directory=ruta_static), name="static")

# =========================================================
# RUTAS DE LA PÁGINA WEB (ENDPOINTS)
# =========================================================

@app.get("/", response_class=HTMLResponse)
async def inicio(request: Request):
    # Extraemos la lista de sesiones únicas disponibles en el dataset
    sesiones_unicas = motor.df['Session'].unique().tolist() if hasattr(motor, 'df') else []
    
    return templates.TemplateResponse(
        request=request, 
        name="index.html", 
        context={"request": request, "resultados": None, "sesiones_disponibles": sesiones_unicas}
    )

@app.get("/buscar", response_class=HTMLResponse)
async def buscar(request: Request, q: str, metodo: str = "llm", sesion: str = None):
    tiempo_inicio = time.time()
    
    # Pasamos el filtro de la sesión al motor
    if metodo == "llm":
        resultados = motor.buscar_llm(q, filtro_sesion=sesion)
    else:
        resultados = motor.buscar_tradicional(q, filtro_sesion=sesion)
        
    tiempo_fin = time.time()
    tiempo_busqueda = round(tiempo_fin - tiempo_inicio, 4)
    sesiones_unicas = motor.df['Session'].unique().tolist()
        
    # Un solo return limpio con todos los datos necesarios
    return templates.TemplateResponse(
        request=request, 
        name="index.html", 
        context={
            "request": request, 
            "query": q,
            "metodo": metodo,
            "sesion_actual": sesion,  # Para saber cuál está seleccionada
            "sesiones_disponibles": sesiones_unicas,
            "resultados": resultados,
            "tiempo_busqueda": tiempo_busqueda
        }
    )

@app.get("/paper/{paper_id}", response_class=HTMLResponse)
async def ver_paper(request: Request, paper_id: int):
    """
    Ruta para ver los detalles completos de un artículo específico.
    """
    paper_data = motor.obtener_paper_por_id(paper_id)
    
    if not paper_data:
        return HTMLResponse(content="<h1>Error 404: Artículo no encontrado</h1>", status_code=404)
        
    return templates.TemplateResponse(
        request=request, 
        name="paper.html", 
        context={"request": request, "paper": paper_data}
    )