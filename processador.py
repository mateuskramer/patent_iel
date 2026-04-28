import os
import json
import psycopg2
from psycopg2 import extras
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
import time
import streamlit as st
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

load_dotenv()

# Configurações do Banco de Dados
LOCAL_DB_CONFIG = {
    'host': os.environ['DB_HOST'],
    'database': os.environ.get('DB_NAME', 'postgres'),
    'user': os.environ['DB_USER'],
    'password': os.environ['DB_PASS'],
    'port': int(os.environ.get('DB_PORT', 5432)),
    'sslmode': 'require'
}

# Dicionário Técnico Completo
SMART_DICTIONARY = {
    "Vehicle Systems": ["vehicle", "car", "automotive", "truck", "bus", "chassis", "bodywork", "frame"],
    "Propulsion & Engine": ["engine", "motor", "combustion", "cylinder", "piston", "fuel system", "transmission", "gearbox"],
    "Electric & Hybrid": ["electric motor", "battery", "ev", "hybrid", "rechargeable", "charging", "inverter", "bms", "lithium"],
    "Driving Assistance & Automation": ["steering", "braking", "abs", "adas", "autonomous", "self-driving", "driverless", "cruise control"],
    "Sensors & Navigation": ["radar", "lidar", "camera", "ultrasonic", "gps", "gnss", "navigation", "mapping", "slam"],
    "Safety & Interior": ["airbag", "seatbelt", "safety", "collision", "dashboard", "hvac", "lighting", "headlamp", "infotainment"],
    "Tires & Suspension": ["tire", "tyre", "wheel", "suspension", "shock absorber", "axle", "rim"],
    "Food Raw Materials": ["food", "beverage", "ingredient", "additive", "protein", "carbohydrate", "lipid", "vitamin"],
    "Food Processing": ["processing", "milling", "grinding", "blending", "mixing", "heating", "cooling", "cooking"],
    "Preservation & Bio": ["preservation", "shelf-life", "pasteurization", "fermentation", "enzyme", "microbiological", "sterilization"],
    "Packaging": ["packaging", "container", "bottle", "can", "wrap", "film", "sealing", "vacuum"],
    "Nutritional & Bioactive": ["nutritional", "probiotic", "supplement", "functional food", "fortified", "antioxidant"],
    "Dairy & Meat Tech": ["milk", "dairy", "cheese", "meat", "poultry", "fish", "plant-based meat"],
    "Quality & Safety": ["haccp", "quality control", "pathogen", "contaminant", "toxicity", "ph level", "moisture"],
    "Fibers & Yarns": ["fiber", "fibre", "yarn", "thread", "filament", "synthetic", "natural fiber", "cotton", "wool", "silk"],
    "Fabric Construction": ["fabric", "textile", "weaving", "knitting", "non-woven", "woven", "braided", "mesh"],
    "Chemical Treatment": ["dyeing", "dye", "pigment", "printing", "finishing", "coating", "bleaching", "impregnation"],
    "Advanced Materials": ["polymer", "composite", "carbon fiber", "nanofiber", "resin", "plastic", "elastomer", "aramid"],
    "Smart & Technical Textiles": ["smart textile", "e-textile", "conductive", "waterproof", "breathable", "flame retardant"],
    "Apparel & Fashion": ["clothing", "apparel", "wearable", "footwear", "sewing", "stitching", "pattern", "tailoring"],
    "Automation & Control": ["automation", "control system", "plc", "scada", "hmi", "controller", "actuator", "valve"],
    "Robotics": ["robot", "robotic", "arm", "manipulator", "end-effector", "cobot", "agv", "drone"],
    "IoT & Digital": ["iot", "internet of things", "sensor", "data", "cloud", "digital twin", "wireless", "rfid", "monitoring"],
    "Manufacturing Processes": ["manufacturing", "production", "assembly", "machining", "tooling", "molding", "casting", "forging"],
    "Additive & 3D": ["3d printing", "additive manufacturing", "prototyping", "layering", "sintering"],
    "Maintenance & Quality": ["maintenance", "predictive", "inspection", "vision system", "testing", "calibration"]
}



@retry(
    stop=stop_after_attempt(3), 
    wait=wait_exponential(multiplier=1, min=4, max=15),
    retry=retry_if_exception_type(Exception)
)
def kickoff_with_retry(crew):
    return crew.kickoff()

def processar_local():
    my_llm = LLM(model='gemini/gemini-2.5-flash', api_key=os.getenv("GEMINI_API_KEY"), temperature=1.0)
    
    conn = psycopg2.connect(**LOCAL_DB_CONFIG)
    cur = conn.cursor(cursor_factory=extras.DictCursor)

    # Busca apenas patentes que ainda não têm termos processados
    cur.execute("""
        SELECT p.id, p.title, p.abstract 
        FROM patents p
        LEFT JOIN patent_terms pt ON p.id = pt.patent_id
        WHERE pt.patent_id IS NULL
    """)
    patentes = cur.fetchall()
    
    if not patentes:
        yield "📭 Sem patentes novas para analisar."
        conn.close()
        return

    agent = Agent(
        role="Patent Specialist",
        goal="Extract technical terms that match the specific categories provided.",
        backstory="Expert in technical classification and patent analysis.",
        verbose=False,
        llm=my_llm
    )

    for pat in patentes:
        # Verifica se o usuário apertou "Parar" no Dashboard
        if st.session_state.get('stop_processing', False):
            yield "🛑 Interrupção detectada. Finalizando..."
            break

        yield f"🧠 Analisando ID: {pat['id']}"
        text = f"{pat['title']}. {pat['abstract']}"
        
        task = Task(
            description=f"""Analyze the text and find matches for these categories: {list(SMART_DICTIONARY.keys())}.
            TEXT: {text}
            RULES: Return ONLY a JSON list. If nothing matches, return empty list.
            FORMAT: {{"topics": [ {{"topic": "category_name"}} ]}}""",
            expected_output="JSON list of matched categories",
            agent=agent,
        )

        crew = Crew(agents=[agent], tasks=[task])
        
        try:
            res = kickoff_with_retry(crew)
            clean_res = res.raw.replace('```json', '').replace('```', '').strip()
            data = json.loads(clean_res)

            topics = data.get("topics", [])
            for item in topics:
                term = item['topic'].strip()
                
                # Garante que o termo existe no dicionário
                cur.execute("""
                    INSERT INTO term_dictionary (term, class, status)
                    VALUES (%s, 'technology', 'approved')
                    ON CONFLICT (term) DO NOTHING RETURNING id
                """, (term,))
                res_id = cur.fetchone()
                term_id = res_id[0] if res_id else None
                
                if not term_id:
                    cur.execute("SELECT id FROM term_dictionary WHERE term = %s", (term,))
                    term_id = cur.fetchone()[0]

                # Associa termo à patente
                cur.execute("INSERT INTO patent_terms (patent_id, term_id) VALUES (%s, %s) ON CONFLICT DO NOTHING", (pat['id'], term_id))
            
            conn.commit()
            yield f"✅ ID {pat['id']} processado ({len(topics)} termos)."
            time.sleep(15) # Cooldown preventivo
            
        except Exception as e:
            conn.rollback()
            yield f"⚠️ Erro no ID {pat['id']}: {str(e)[:50]}..."
            time.sleep(5)

    conn.close()
    yield "🏁 Fim do processamento."