import os
import json
import psycopg2
from psycopg2 import extras
from dotenv import load_dotenv
import google.generativeai as genai
import time
import streamlit as st
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

load_dotenv()

DB_CONFIG = {
    'host': os.environ['DB_HOST'],
    'database': os.environ.get('DB_NAME', 'postgres'),
    'user': os.environ['DB_USER'],
    'password': os.environ['DB_PASS'],
    'port': int(os.environ.get('DB_PORT', 5432)),
    'sslmode': 'require'
}

SMART_DICTIONARY = {
    # --- GRUPO: AUTOMOTIVO E MOBILIDADE (A-Z do Veículo) ---
    "Vehicle Systems": ["vehicle", "car", "automotive", "truck", "bus", "lorry", "terrestrial vehicle", "chassis", "bodywork", "frame"],
    "Propulsion & Engine": ["engine", "motor", "combustion", "cylinder", "piston", "fuel system", "exhaust", "transmission", "gearbox", "clutch", "drivetrain"],
    "Electric & Hybrid": ["electric motor", "battery", "ev", "hybrid", "rechargeable", "charging", "inverter", "stator", "rotor", "bms", "lithium"],
    "Driving Assistance & Automation": ["steering", "braking", "abs", "adas", "autonomous", "self-driving", "driverless", "cruise control", "lane", "parking"],
    "Sensors & Navigation": ["radar", "lidar", "camera", "ultrasonic", "gps", "gnss", "navigation", "mapping", "slam", "odometry"],
    "Safety & Interior": ["airbag", "seatbelt", "safety", "collision", "dashboard", "hvac", "lighting", "headlamp", "infotainment"],
    "Tires & Suspension": ["tire", "tyre", "wheel", "suspension", "shock absorber", "axle", "rim"],

    # --- GRUPO: ALIMENTAR, AGRO E QUÍMICA (Do Campo à Mesa) ---
    "Food Raw Materials": ["food", "beverage", "ingredient", "additive", "protein", "carbohydrate", "lipid", "vitamin", "extract", "essence"],
    "Food Processing": ["processing", "milling", "grinding", "blending", "mixing", "heating", "cooling", "cooking", "frying", "baking", "extrusion"],
    "Preservation & Bio": ["preservation", "shelf-life", "pasteurization", "fermentation", "enzyme", "microbiological", "antimicrobial", "sterilization"],
    "Packaging": ["packaging", "container", "bottle", "can", "wrap", "film", "sealing", "vacuum", "labeling", "package"],
    "Nutritional & Bioactive": ["nutritional", "probiotic", "supplement", "functional food", "fortified", "antioxidant", "flavonoid"],
    "Dairy & Meat Tech": ["milk", "dairy", "cheese", "meat", "poultry", "fish", "plant-based meat", "alternative protein"],
    "Quality & Safety": ["haccp", "quality control", "pathogen", "contaminant", "toxicity", "ph level", "moisture"],

    # --- GRUPO: TÊXTIL E CIÊNCIA DOS MATERIAIS (Da Fibra ao Tecido) ---
    "Fibers & Yarns": ["fiber", "fibre", "yarn", "thread", "filament", "synthetic", "natural fiber", "cotton", "wool", "silk", "polyester", "nylon", "acrylic"],
    "Fabric Construction": ["fabric", "textile", "weaving", "knitting", "non-woven", "woven", "braided", "mesh", "cloth", "garment"],
    "Chemical Treatment": ["dyeing", "dye", "pigment", "printing", "finishing", "coating", "bleaching", "scouring", "impregnation"],
    "Advanced Materials": ["polymer", "composite", "carbon fiber", "nanofiber", "resin", "plastic", "elastomer", "aramid", "kevlar"],
    "Smart & Technical Textiles": ["smart textile", "e-textile", "conductive", "waterproof", "breathable", "flame retardant", "protective clothing"],
    "Apparel & Fashion": ["clothing", "apparel", "wearable", "footwear", "sewing", "stitching", "pattern", "tailoring"],

    # --- GRUPO: INDÚSTRIA 4.0 E MANUFATURA (O "Cérebro" da Fábrica) ---
    "Automation & Control": ["automation", "control system", "plc", "scada", "hmi", "controller", "actuator", "valve", "pneumatic", "hydraulic"],
    "Robotics": ["robot", "robotic", "arm", "manipulator", "end-effector", "cobot", "agv", "drone"],
    "IoT & Digital": ["iot", "internet of things", "sensor", "data", "cloud", "digital twin", "wireless", "rfid", "monitoring"],
    "Manufacturing Processes": ["manufacturing", "production", "assembly", "machining", "tooling", "molding", "casting", "forging", "stamping"],
    "Additive & 3D": ["3d printing", "additive manufacturing", "prototyping", "layering", "sintering"],
    "Maintenance & Quality": ["maintenance", "predictive", "inspection", "vision system", "testing", "calibration"]
}

CATEGORIES_TEXT = "\n".join([
    f"- {category}\n  Keywords: {', '.join(keywords)}"
    for category, keywords in SMART_DICTIONARY.items()
])

PROMPT_TEMPLATE = """You are a patent classification expert.
Analyze the patent text and select ONLY the categories strongly supported by the patent content.
Avoid broad or weak matches — only include a category if the patent clearly and directly relates to it.

AVAILABLE CATEGORIES AND KEYWORDS:
{categories}

PATENT TEXT:
{text}

RULES:
- Return ONLY exact category names from the list above.
- A category must be strongly supported by the patent text to be included.
- Do NOT include categories with only weak or tangential connections.
- Return ONLY a JSON object — no markdown, no explanation, no preamble.
- If nothing matches, return {{"topics": []}}.

FORMAT:
{{"topics": [{{"topic": "Category Name"}}, {{"topic": "Another Category"}}]}}"""


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=15),
    retry=retry_if_exception_type(Exception)
)
def classify_with_retry(model, text):
    prompt = PROMPT_TEMPLATE.format(categories=CATEGORIES_TEXT, text=text)
    response = model.generate_content(prompt)
    return response.text


def processar_local():
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        generation_config=genai.GenerationConfig(
            temperature=0.1,
            top_p=0.1,
            max_output_tokens=2048,
        )
    )

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor(cursor_factory=extras.DictCursor)

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

    for pat in patentes:
        if st.session_state.get('stop_processing', False):
            yield "🛑 Interrupção detectada. Finalizando..."
            break

        yield f"🧠 Analisando ID: {pat['id']}"
        text = f"{pat['title']}. {pat['abstract']}"

        try:
            raw = classify_with_retry(model, text)
            clean = raw.replace('```json', '').replace('```', '').strip()
            data = json.loads(clean)
            topics = data.get("topics", [])

            for item in topics:
                term = item['topic'].strip().lower
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

                cur.execute(
                    "INSERT INTO patent_terms (patent_id, term_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                    (pat['id'], term_id)
                )

            conn.commit()
            yield f"✅ ID {pat['id']} processado ({len(topics)} termos)."
            time.sleep(15)

        except Exception as e:
            conn.rollback()
            yield f"⚠️ Erro no ID {pat['id']}: {str(e)[:50]}..."
            time.sleep(5)

    conn.close()
    yield "🏁 Fim do processamento."