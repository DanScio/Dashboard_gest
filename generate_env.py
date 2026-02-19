from pathlib import Path
import json

# ==============================
# CONFIGURAZIONE BASE (COMUNE)
# ==============================
BASE_ENV_CONTENT = """\
DB_HOST=localhost
DB_PORT=5432
DB_NAME=dashboard_gest
DB_SSLMODE=disable
"""

# DEMO_MODE per avvio locale (0/1)
DEFAULT_DEMO_MODE = "1"

# Cartella env reale del progetto
ENV_DIR = Path(__file__).resolve().parent / "env"
ENV_DIR.mkdir(parents=True, exist_ok=True)

# ==============================
# CREDENZIALI UI (NEGOZI)
# ==============================
# Streamlit si collega con app_ui e poi fa SET ROLE = op_*
APP_UI_USER = "app_ui"
APP_UI_PASSWORD = "619"

NEGOZI = {
    "ancona_centro": "op_ancona_centro",
    "conero": "op_conero",
    "chieti_scalo": "op_chieti_scalo",
    "santangelo": "op_santangelo",
    "val_vibrata": "op_val_vibrata",
    "san_giovanni_teatino": "op_san_giovanni_teatino",
    "porto_grande": "op_porto_grande",
    "vasto": "op_vasto",
    "campobasso": "op_campobasso",
    "pescara_veii": "op_pescara_veii",
}

# Password negozi (login UI)
NEGOZI_PASSWORD = {
    "op_ancona_centro": "Ancona#24",
    "op_conero": "Conero#24",
    "op_chieti_scalo": "Chieti#24",
    "op_santangelo": "Sangel#24",
    "op_val_vibrata": "Valvib#24",
    "op_san_giovanni_teatino": "SGTeat#24",
    "op_porto_grande": "Porto#24",
    "op_vasto": "Vasto#24",
    "op_campobasso": "Campo#24",
    "op_pescara_veii": "Pesca#24",
}

# ==============================
# PROFILI EXTRA (facoltativi)
# ==============================
GENERATE_APP_OWNER = True
APP_OWNER_USER = "app_owner"
APP_OWNER_PASSWORD = "619"

GENERATE_DASHBOARD_OWNER = True
DASHBOARD_OWNER_USER = "dashboard_owner"
DASHBOARD_OWNER_PASSWORD = "619"

GENERATE_IMPORT_APP_OWNER = True
IMPORT_USER = "app_owner"
IMPORT_PASSWORD = "Danilo33"

# ==============================
# USERS_JSON (per login locale)
# ==============================
def build_users_json() -> str:
    """
    JSON compatto (una riga) user->password:
    USERS_JSON={"app_owner":"619", ...}
    """
    users: dict[str, str] = {}

    if GENERATE_APP_OWNER:
        users[APP_OWNER_USER] = APP_OWNER_PASSWORD
    if GENERATE_DASHBOARD_OWNER:
        users[DASHBOARD_OWNER_USER] = DASHBOARD_OWNER_PASSWORD

    for _, role in NEGOZI.items():
        pwd = NEGOZI_PASSWORD.get(role)
        if pwd:
            users[role] = pwd

    return json.dumps(users, ensure_ascii=False, separators=(",", ":"))


USERS_JSON_VALUE = build_users_json()

# ==============================
# HELPERS
# ==============================
def write_env(filename: str, db_user: str, db_password: str, db_role: str) -> None:
    env_path = ENV_DIR / filename
    content = (
        BASE_ENV_CONTENT
        + f"DB_USER={db_user}\n"
        + f"DB_PASSWORD={db_password}\n"
        + f"DB_ROLE={db_role}\n"
        + f"USERS_JSON={USERS_JSON_VALUE}\n"
        + f"DEMO_MODE={DEFAULT_DEMO_MODE}\n"
    )
    env_path.write_text(content, encoding="utf-8")
    print(f"Creato/Aggiornato: env\\{filename}")


# ==============================
# GENERAZIONE
# ==============================
def generate_store_env_files() -> None:
    for negozio, role in NEGOZI.items():
        filename = f".env_{negozio}"
        write_env(filename, APP_UI_USER, APP_UI_PASSWORD, role)


def generate_optional_profiles() -> None:
    if GENERATE_APP_OWNER:
        write_env(".env_app_owner", APP_OWNER_USER, APP_OWNER_PASSWORD, "app_owner")

    if GENERATE_DASHBOARD_OWNER:
        write_env(".env_dashboard_owner", DASHBOARD_OWNER_USER, DASHBOARD_OWNER_PASSWORD, "dashboard_owner")

    if GENERATE_IMPORT_APP_OWNER:
        write_env(".env_import_app_owner", IMPORT_USER, IMPORT_PASSWORD, "app_owner")


if __name__ == "__main__":
    generate_store_env_files()
    generate_optional_profiles()
