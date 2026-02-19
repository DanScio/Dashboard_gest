from pathlib import Path

# ==============================
# CONFIGURAZIONE BASE (COMUNE)
# ==============================
BASE_ENV_CONTENT = """\
DB_HOST=localhost
DB_PORT=5432
DB_NAME=dashboard_gest
DB_SSLMODE=disable
"""

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

# ==============================
# PROFILI EXTRA (facoltativi)
# ==============================
# Se vuoi rigenerare anche questi, valorizza le password e metti GENERATE_* = True
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
# HELPERS
# ==============================
def write_env(filename: str, db_user: str, db_password: str, db_role: str) -> None:
    env_path = ENV_DIR / filename
    content = (
        BASE_ENV_CONTENT
        + f"DB_USER={db_user}\n"
        + f"DB_PASSWORD={db_password}\n"
        + f"DB_ROLE={db_role}\n"
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
        # profilo import: DB_USER=app_owner e DB_ROLE=app_owner NON serve, perché per import usi login diretto app_owner.
        # qui teniamo DB_ROLE coerente, se poi lo vuoi diverso lo modifichiamo.
        write_env(".env_import_app_owner", IMPORT_USER, IMPORT_PASSWORD, "app_owner")


if __name__ == "__main__":
    generate_store_env_files()
    generate_optional_profiles()
