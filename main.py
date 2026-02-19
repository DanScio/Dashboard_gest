import os
from datetime import datetime, timedelta
from pathlib import Path
import random

import pandas as pd
import psycopg
from psycopg import sql as psql
from dotenv import load_dotenv
import streamlit as st
import plotly.graph_objects as go

# Click sulle torte (opzionale; se non installato, fallback senza drill-click)
try:
    from streamlit_plotly_events import plotly_events  # pip install streamlit-plotly-events
except Exception:
    plotly_events = None


# -------------------------
# ENV (.env + DOTENV_PATH) - in locale
# (su Streamlit Cloud userai st.secrets, ma lasciamo invariato)
# -------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
dotenv_path = os.getenv("DOTENV_PATH")
if dotenv_path:
    p = Path(dotenv_path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    load_dotenv(p, override=True)
else:
    load_dotenv(PROJECT_ROOT / ".env", override=True)

st.set_page_config(page_title="Dashboard Gest", layout="wide")


# -------------------------
# Secrets / Demo / Login (Streamlit Cloud)
# -------------------------
def is_demo_mode() -> bool:
    try:
        return str(st.secrets.get("DEMO_MODE", "0")) == "1"
    except Exception:
        return os.getenv("DEMO_MODE", "0") == "1"


def get_users_dict() -> dict[str, str]:
    try:
        users = st.secrets.get("users", {})
        return {str(k): str(v) for k, v in users.items()}
    except Exception:
        return {}


def render_login() -> None:
    """
    Login definitivo: username + password.
    Il ruolo DB coincide con lo username (app_owner / dashboard_owner / op_*).
    """
    if st.session_state.get("auth_ok") is True and st.session_state.get("db_role"):
        return

    st.title("Dashboard Gest")
    st.caption("Accesso")

    users = get_users_dict()
    if not users:
        st.error("Credenziali non configurate (Settings → Secrets → [users]).")
        st.stop()

    with st.form("login_form", clear_on_submit=False):
        u = st.text_input("Utente").strip()
        p = st.text_input("Password", type="password")
        ok = st.form_submit_button("Entra", type="primary")

    if not ok:
        st.stop()

    if not u or users.get(u) != p:
        st.error("Credenziali non valide.")
        st.stop()

    st.session_state["auth_ok"] = True
    st.session_state["db_role"] = u
    st.rerun()


with st.expander("Debug config (.env / secrets)"):
    st.write("DEMO_MODE:", "1" if is_demo_mode() else "0")
    st.write("DOTENV_PATH:", os.getenv("DOTENV_PATH"))
    st.write("DB_HOST:", os.getenv("DB_HOST"))
    st.write("DB_PORT:", os.getenv("DB_PORT"))
    st.write("DB_NAME:", os.getenv("DB_NAME"))
    st.write("DB_USER:", os.getenv("DB_USER"))
    st.write("DB_PASSWORD presente:", bool(os.getenv("DB_PASSWORD")))
    st.write("DB_DSN/DATABASE_URL presente:", bool(os.getenv("DB_DSN") or os.getenv("DATABASE_URL")))
    st.write("DB_ROLE (.env):", os.getenv("DB_ROLE") or "app_readonly")
    try:
        st.write("Secrets users presenti:", bool(st.secrets.get("users", {})))
    except Exception:
        st.write("Secrets users presenti:", False)


# -------------------------
# DB config
# -------------------------
def get_db_dsn() -> str:
    # Streamlit Cloud: puoi anche mettere DB_DSN in secrets e leggerlo qui se vuoi in futuro.
    dsn = os.getenv("DB_DSN") or os.getenv("DATABASE_URL")
    if dsn:
        return dsn

    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    name = os.getenv("DB_NAME")
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")

    missing = [k for k, v in {"DB_NAME": name, "DB_USER": user, "DB_PASSWORD": password}.items() if not v]
    if missing:
        raise RuntimeError(f"Variabili .env mancanti: {', '.join(missing)}")

    return f"host={host} port={port} dbname={name} user={user} password={password}"


def get_db_role() -> str:
    # Se c'è login, il ruolo arriva dalla sessione
    if st.session_state.get("auth_ok") is True and st.session_state.get("db_role"):
        return str(st.session_state["db_role"])
    return os.getenv("DB_ROLE") or "app_readonly"


def is_store_role(db_role: str) -> bool:
    return (db_role or "").startswith("op_")


def is_dashboard_role(db_role: str) -> bool:
    return db_role in ("app_owner", "dashboard_owner")


# -------------------------
# DEMO DATA (mock ricchi)
# -------------------------
def _demo_seed() -> int:
    # seed stabile per sessione (così non cambia a ogni rerun)
    if "demo_seed" not in st.session_state:
        st.session_state["demo_seed"] = random.randint(10_000, 99_999)
    return int(st.session_state["demo_seed"])


def _demo_now() -> datetime:
    return datetime.now()


def demo_query(sql: str, params: tuple | None, db_role: str) -> list[dict]:
    """
    Mock "ricchi" coerenti con la UI.
    """
    _ = _demo_seed()
    s = (sql or "").lower()

    # elenco punti vendita
    if "app.api_elenco_punti_vendita" in s:
        return [
            {"punto_vendita_id": 7, "punto_vendita": "Ancona Centro"},
            {"punto_vendita_id": 8, "punto_vendita": "Conero"},
            {"punto_vendita_id": 9, "punto_vendita": "Chieti Scalo"},
        ]

    # mapping piste (DEMO)
    if "app.api_pista_map" in s:
        return [
            {"codice": "MNP", "valore_db": "A.2 VOLUMI SIM MNP"},
            {"codice": "FAMILY", "valore_db": "A.5 VOLUMI CONVERGENZA FISSO - MOBILE"},
        ]

    # KPI torte (api_kpi_esiti)
    if "app.api_kpi_esiti" in s:
        # piccolo shake basato sul ruolo (solo estetico)
        bump = 0
        if db_role.startswith("op_"):
            bump = 7
        return [
            {"stato": "DA_ESITARE", "totale": 140 + bump},
            {"stato": "IN_LAVORAZIONE", "totale": 18 + (bump // 2)},
            {"stato": "OK", "totale": 64 + bump},
            {"stato": "KO", "totale": 22 + (bump // 3)},
            {"stato": "BO_OK", "totale": 9},
            {"stato": "BO_KO", "totale": 4},
        ]

    # drill-down (api_drill_esiti)
    if "app.api_drill_esiti" in s:
        # params: (selected_state, pv_id, selected_pista_for_drill, limit)
        stato = params[0] if params and len(params) >= 1 else "DA_ESITARE"
        pista = params[2] if params and len(params) >= 3 else None
        limit = int(params[3]) if params and len(params) >= 4 else 300
        limit = max(1, min(limit, 5000))

        pv_map = {7: "Ancona Centro", 8: "Conero", 9: "Chieti Scalo"}
        pv_id = params[1] if params and len(params) >= 2 else None
        pv_name = pv_map.get(int(pv_id), "Ancona Centro") if pv_id else "TUTTI"

        base_date = _demo_now().date() - timedelta(days=25)
        rows: list[dict] = []
        for i in range(1, min(limit, 200) + 1):
            att_id = 1000 + i
            d_att = datetime.combine(base_date + timedelta(days=(i % 25)), datetime.min.time()) + timedelta(hours=9)
            esito_at = d_att + timedelta(days=1, hours=1)

            if pista is None:
                pista_val = "A.2 VOLUMI SIM MNP" if i % 2 == 0 else "A.5 VOLUMI CONVERGENZA FISSO - MOBILE"
            else:
                pista_val = pista

            nota = "Nota demo"
            if str(stato) == "IN_LAVORAZIONE":
                nota = "In lavorazione: richiesta integrazione documentale"

            rows.append(
                {
                    "attivazione_id": att_id,
                    "data_attivazione": d_att,
                    "telefono": f"349{_demo_seed():05d}{i:03d}"[-10:],
                    "seriale_sim": f"SIM-DEMO-{i:06d}",
                    "pista": pista_val,
                    "esito_corrente": None if str(stato) == "DA_ESITARE" else str(stato),
                    "esito_at": esito_at if str(stato) != "DA_ESITARE" else None,
                    "nota": nota if str(stato) != "DA_ESITARE" else None,
                    "punto_vendita": pv_name if pv_name != "TUTTI" else (pv_map[7] if i % 3 == 0 else pv_map[8]),
                }
            )
        return rows

    # Worklist store mode (v_pratiche_lista_sec)
    if "from app.v_pratiche_lista_sec" in s:
        pista = params[0] if params and len(params) >= 1 else "A.2 VOLUMI SIM MNP"
        base_date = _demo_now().date() - timedelta(days=20)

        rows: list[dict] = []
        for i in range(1, 121):  # worklist ricca
            att_id = 2000 + i
            d_att = datetime.combine(base_date + timedelta(days=(i % 20)), datetime.min.time()) + timedelta(hours=10)
            # un po' di IN_LAVORAZIONE sparsi
            esito_corr = "IN_LAVORAZIONE" if i % 17 == 0 else None
            nota = "Attendere esito cliente" if esito_corr == "IN_LAVORAZIONE" else None

            # next_action_label coerente
            if esito_corr == "IN_LAVORAZIONE":
                label = "Rilavorare: completa pratica"
            else:
                label = "Esita la pratica (OK/KO)"

            rows.append(
                {
                    "attivazione_id": att_id,
                    "data_attivazione": d_att,
                    "telefono": f"348{_demo_seed():05d}{i:03d}"[-10:],
                    "seriale_sim": f"WLSIM-{i:06d}",
                    "pista": pista,
                    "esito_corrente": esito_corr,
                    "nota": nota,
                    "next_action_label": label,
                }
            )

        # simuliamo ordering “vecchio → nuovo” se la query contiene "order by wl.data_attivazione asc"
        if "order by wl.data_attivazione asc" in s:
            rows.sort(key=lambda r: (r["data_attivazione"], r["attivazione_id"]))
        else:
            # default: simula sort_at asc (qui la data_attivazione già fa il lavoro)
            rows.sort(key=lambda r: (r["data_attivazione"], r["attivazione_id"]))
        return rows[:500]

    # KPI tab (v_attivazione_stato_sec group by)
    if "from app.v_attivazione_stato_sec" in s and "group by 1" in s:
        return [
            {"stato": "DA_ESITARE", "totale": 92},
            {"stato": "IN_LAVORAZIONE", "totale": 11},
            {"stato": "OK", "totale": 37},
            {"stato": "KO", "totale": 14},
            {"stato": "BO_OK", "totale": 6},
            {"stato": "BO_KO", "totale": 3},
        ]

    # pratiche lavorate (v_attivazione_stato_sec con esito_corrente not null)
    if "from app.v_attivazione_stato_sec" in s and "where s.esito_corrente is not null" in s:
        pista = params[0] if params and len(params) >= 1 else "A.2 VOLUMI SIM MNP"
        base_date = _demo_now() - timedelta(days=14)
        rows: list[dict] = []
        for i in range(1, 51):
            d_att = base_date - timedelta(days=(i % 14))
            rows.append(
                {
                    "data_attivazione": d_att,
                    "telefono": f"3497777{i:03d}",
                    "seriale_sim": f"SIM-WORKED-{i:06d}",
                    "pista": pista,
                    "esito_corrente": "OK" if i % 5 != 0 else "KO",
                    "esito_at": d_att + timedelta(hours=2),
                    "nota": "Esempio nota" if i % 5 == 0 else None,
                }
            )
        # in UI mostri solo alcune colonne
        return rows

    # chiamate di scrittura (DEMO): simuliamo event_id sempre
    if "app.api_take_in_charge" in s or "app.api_ins_esito_app" in s or "app.api_ins_esito_bo" in s:
        return [{"event_id": 999999}]

    # debug identity
    if "select session_user" in s and "current_user" in s:
        return [{"session_user": "streamlit_demo", "current_user": db_role}]

    return []


# -------------------------
# DB (cache + SET ROLE immediato)
# -------------------------
@st.cache_data(ttl=15)
def query(sql: str, params: tuple | None, db_role: str) -> list[dict]:
    if is_demo_mode():
        return demo_query(sql, params, db_role)

    dsn = get_db_dsn()
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur_role:
            cur_role.execute(psql.SQL("SET ROLE {}").format(psql.Identifier(db_role)))

        with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            cur.execute(sql, params or ())
            return cur.fetchall()


def q(sql: str, params: tuple | None = None) -> list[dict]:
    return query(sql, params, get_db_role())


# -------------------------
# Helpers DB-first (funzioni esistenti)
# -------------------------
def take_in_charge(attivazione_id: int, note: str) -> int:
    res = q("SELECT app.api_take_in_charge(%s, %s) AS event_id;", (attivazione_id, note))
    return int(res[0]["event_id"])


def close_app(attivazione_id: int, valore: str, note: str | None) -> int:
    res = q(
        "SELECT app.api_ins_esito_app(%s, %s::app.esito_valore, %s) AS event_id;",
        (attivazione_id, valore, note),
    )
    return int(res[0]["event_id"])


def close_bo(attivazione_id: int, valore: str, note: str | None) -> int:
    res = q(
        "SELECT app.api_ins_esito_bo(%s, %s::app.esito_valore, %s) AS event_id;",
        (attivazione_id, valore, note),
    )
    return int(res[0]["event_id"])


# -------------------------
# Mapping piste (DB-side)
# -------------------------
@st.cache_data(ttl=15)
def get_pista_map(db_role: str) -> dict[str, str]:
    rows = query("SELECT codice, valore_db FROM app.api_pista_map();", None, db_role)
    return {str(r["codice"]): str(r["valore_db"]) for r in rows}


def pista_value(codice: str) -> str:
    m = get_pista_map(get_db_role())
    if codice not in m:
        raise RuntimeError(f"Pista non mappata in app.pista_map: {codice}")
    return m[codice]


# -------------------------
# Colori esiti (coerenti)
# -------------------------
ESITO_COLORS = {
    "DA_ESITARE": "#9CA3AF",      # grigio
    "IN_LAVORAZIONE": "#F59E0B",  # ambra
    "OK": "#10B981",              # verde
    "KO": "#EF4444",              # rosso
    "BO_OK": "#3B82F6",           # blu
    "BO_KO": "#8B5CF6",           # viola
}
DARK_BG = "rgb(14,17,24)"


# -------------------------
# LOGIN (prima di usare db_role)
# -------------------------
render_login()

# -------------------------
# UI
# -------------------------
db_role = get_db_role()
store_mode = is_store_role(db_role)
dashboard_mode = is_dashboard_role(db_role)

st.title("Dashboard Gest")
st.caption(f"Aggiornato: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")

col_r, col_logout = st.columns([1, 1])
with col_r:
    if st.button("Aggiorna dati"):
        st.cache_data.clear()
        st.rerun()
with col_logout:
    if st.button("Logout"):
        st.session_state.clear()
        st.cache_data.clear()
        st.rerun()

if os.getenv("APP_DEBUG") == "1":
    with st.expander("Debug DB identity (session_user/current_user)"):
        ident = q("SELECT session_user::text AS session_user, current_user::text AS current_user;", None)
        st.write(ident[0] if ident else {})

st.divider()

if store_mode:
    st.info("Modalità negozio: lavora le pratiche direttamente dalla Worklist operativa (DB-side).")
elif dashboard_mode:
    st.info("Modalità dashboard: KPI a torta + drill-down (DB-first).")
else:
    st.warning("Ruolo DB non riconosciuto per questa UI.")
    st.stop()


# ==========================
# DASHBOARD (app_owner / dashboard_owner)
# ==========================
def build_pie_figure(df_kpi: pd.DataFrame, title: str) -> tuple[go.Figure, list[str]]:
    df = df_kpi.copy()
    df["stato"] = df["stato"].astype(str)
    df["totale"] = pd.to_numeric(df["totale"], errors="coerce").fillna(0).astype(int)
    df = df[df["totale"] > 0].copy()
    df = df.sort_values("totale", ascending=False)

    labels = df["stato"].tolist()
    values = df["totale"].tolist()
    colors = [ESITO_COLORS.get(s, "#9CA3AF") for s in labels]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.45,
                marker=dict(colors=colors),
                texttemplate="%{label}<br>%{value} (%{percent:.2%})",
                hovertemplate="%{label}<br>Totale: %{value}<br>Percent: %{percent:.4%}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title=title,
        margin=dict(l=20, r=20, t=60, b=20),
        legend_title_text="Esito",
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_BG,
        font=dict(color="white"),
    )
    return fig, labels


def render_one_pie(pv_id: int | None, p_pista: str | None, title: str, key_suffix: str) -> tuple[str | None, str | None]:
    rows = q("SELECT stato, totale FROM app.api_kpi_esiti(%s, %s);", (pv_id, p_pista))
    if not rows:
        st.info(f"Nessun dato per: {title}")
        return None, None

    df_kpi = pd.DataFrame(rows)
    fig, labels = build_pie_figure(df_kpi, title=title)

    clicked_state = None
    if plotly_events is not None:
        clicked = plotly_events(
            fig,
            click_event=True,
            hover_event=False,
            select_event=False,
            override_height=420,
            key=f"pie_{key_suffix}",
        )
        if clicked and isinstance(clicked, list) and len(clicked) > 0:
            idx = int(clicked[0].get("pointNumber", -1))
            if 0 <= idx < len(labels):
                clicked_state = str(labels[idx])
    else:
        st.plotly_chart(fig, width="stretch")
        st.caption("Per abilitare il click sulle torte: `pip install streamlit-plotly-events`")

    if clicked_state:
        return clicked_state, p_pista
    return None, None


def render_dashboard_esiti() -> None:
    st.subheader("Dashboard Esiti")

    pv_rows = q("SELECT punto_vendita_id, punto_vendita FROM app.api_elenco_punti_vendita();")
    pv_options = [("TUTTI", None)] + [(r["punto_vendita"], int(r["punto_vendita_id"])) for r in pv_rows]
    pv_label_to_id = {lbl: pv_id for (lbl, pv_id) in pv_options}

    pista_options = [
        ("TUTTE", None),
        ("MNP", "MNP"),
        ("FAMILY", "FAMILY"),
    ]
    pista_label_to_code = {lbl: v for (lbl, v) in pista_options}

    colA, colB, colC = st.columns([2, 2, 2])
    with colA:
        pv_label = st.selectbox("Negozio", options=[lbl for (lbl, _) in pv_options], index=0)
    with colB:
        pista_label = st.selectbox("Pista", options=[lbl for (lbl, _) in pista_options], index=0)
    with colC:
        limit = st.number_input("Righe drill-down (max 5000)", min_value=10, max_value=5000, value=300, step=10)

    pv_id = pv_label_to_id[pv_label]
    pista_code = pista_label_to_code[pista_label]
    pista_db_value = pista_value(pista_code) if pista_code is not None else None

    selected_state = None
    selected_pista_for_drill = None

    if pista_db_value is None:
        st.caption("Pista: TUTTE → torte per pista (MNP + FAMILY)")
        c1, c2 = st.columns(2)

        with c1:
            s, psel = render_one_pie(pv_id, pista_value("MNP"), "MNP • Distribuzione esiti", "mnp")
            if s:
                selected_state, selected_pista_for_drill = s, psel

        with c2:
            s, psel = render_one_pie(pv_id, pista_value("FAMILY"), "FAMILY • Distribuzione esiti", "family")
            if s:
                selected_state, selected_pista_for_drill = s, psel

    else:
        s, psel = render_one_pie(pv_id, pista_db_value, "Distribuzione esiti", "single")
        if s:
            selected_state, selected_pista_for_drill = s, psel

    st.divider()
    st.subheader("Dettaglio (drill-down)")

    if not selected_state:
        st.info("Clicca una fetta della torta per vedere le righe associate.")
        return

    st.caption(f"Selezionato: **{selected_state}** • Pista: **{(selected_pista_for_drill or 'TUTTE')}**")

    drill_rows = q(
        "SELECT attivazione_id, data_attivazione, telefono, seriale_sim, pista, esito_corrente, esito_at, nota, punto_vendita "
        "FROM app.api_drill_esiti(%s, %s, %s, %s);",
        (selected_state, pv_id, selected_pista_for_drill, int(limit)),
    )

    if not drill_rows:
        st.info("Nessuna riga per lo stato selezionato.")
        return

    df_drill = pd.DataFrame(drill_rows)

    if "data_attivazione" in df_drill.columns:
        df_drill["data_attivazione"] = pd.to_datetime(df_drill["data_attivazione"], errors="coerce").dt.strftime("%d/%m/%Y")
    if "esito_at" in df_drill.columns:
        df_drill["esito_at"] = pd.to_datetime(df_drill["esito_at"], errors="coerce").dt.strftime("%d/%m/%Y %H:%M:%S")

    if "drill_sel_reset_token" not in st.session_state:
        st.session_state["drill_sel_reset_token"] = 0

    if get_db_role() == "app_owner":
        df_edit = df_drill.copy().set_index("attivazione_id")
        df_edit.insert(0, "SEL", False)

        edited = st.data_editor(
            df_edit,
            key=f"drill_editor_{st.session_state['drill_sel_reset_token']}",
            hide_index=True,
            width="stretch",
            height=420,
            disabled=[c for c in df_edit.columns if c != "SEL"],
            column_config={
                "SEL": st.column_config.CheckboxColumn("SEL"),
                "data_attivazione": st.column_config.TextColumn("Data attivazione"),
                "telefono": st.column_config.TextColumn("Telefono"),
                "seriale_sim": st.column_config.TextColumn("Seriale SIM"),
                "pista": st.column_config.TextColumn("Pista"),
                "esito_corrente": st.column_config.TextColumn("Esito"),
                "esito_at": st.column_config.TextColumn("Esito At"),
                "nota": st.column_config.TextColumn("Nota"),
                "punto_vendita": st.column_config.TextColumn("Negozio"),
            },
        )

        sel = edited[edited["SEL"] == True]  # noqa: E712
        if len(sel) == 1:
            selected_att_id = int(sel.index[0])
            row = sel.iloc[0].to_dict()

            st.success(
                "Riga selezionata"
                f"\n\n- attivazione_id: {selected_att_id}"
                f"\n- Data: {row.get('data_attivazione')}"
                f"\n- Telefono: {row.get('telefono')}"
                f"\n- Seriale SIM: {row.get('seriale_sim')}"
                f"\n- Pista: {row.get('pista')}"
                f"\n- Esito attuale: {row.get('esito_corrente')}"
            )

            st.session_state["selected_pratica"] = {
                "attivazione_id": selected_att_id,
                "data_attivazione": row.get("data_attivazione"),
                "telefono": row.get("telefono"),
                "seriale_sim": row.get("seriale_sim"),
                "pista": row.get("pista"),
                "esito_corrente": row.get("esito_corrente"),
                "nota_attuale": row.get("nota"),
                "punto_vendita": row.get("punto_vendita"),
            }
        elif len(sel) > 1:
            st.warning("Seleziona UNA sola riga (SEL).")
            st.session_state.pop("selected_pratica", None)
        else:
            st.session_state.pop("selected_pratica", None)

    else:
        st.dataframe(
            df_drill,
            width="stretch",
            hide_index=True,
            height=420,
            column_config={
                "attivazione_id": st.column_config.NumberColumn("ID"),
                "data_attivazione": st.column_config.TextColumn("Data attivazione"),
                "telefono": st.column_config.TextColumn("Telefono"),
                "seriale_sim": st.column_config.TextColumn("Seriale SIM"),
                "pista": st.column_config.TextColumn("Pista"),
                "esito_corrente": st.column_config.TextColumn("Esito"),
                "esito_at": st.column_config.TextColumn("Esito At"),
                "nota": st.column_config.TextColumn("Nota"),
                "punto_vendita": st.column_config.TextColumn("Negozio"),
            },
        )

    # Gestione esiti (solo app_owner)
    if get_db_role() == "app_owner":
        st.divider()
        st.subheader("Gestione esiti (app_owner)")

        pratica = st.session_state.get("selected_pratica") or {}
        default_id = int(pratica.get("attivazione_id") or 1)
        default_note = str(pratica.get("nota_attuale") or "").strip()

        with st.form("form_esiti_app_owner", clear_on_submit=False):
            col1, col2 = st.columns([1, 2])
            with col1:
                att_id = st.number_input("attivazione_id", min_value=1, step=1, value=default_id)
            with col2:
                esito_new = st.selectbox(
                    "Esito da applicare",
                    options=["IN_LAVORAZIONE", "OK", "KO", "BO_OK", "BO_KO"],
                    index=1,
                )

            if pratica:
                st.caption(
                    f"Contesto selezionato → "
                    f"Telefono: **{pratica.get('telefono')}** • "
                    f"Seriale SIM: **{pratica.get('seriale_sim')}** • "
                    f"Pista: **{pratica.get('pista')}** • "
                    f"Esito attuale: **{pratica.get('esito_corrente')}**"
                )

            nota = st.text_area("Nota", height=110, max_chars=500, value=default_note)
            apply_btn = st.form_submit_button("Applica esito", type="primary")

        if apply_btn:
            att_id = int(att_id)
            esito_new = str(esito_new).strip()
            nota_txt = str(nota or "").strip()

            if esito_new == "IN_LAVORAZIONE" and not nota_txt:
                st.error("IN_LAVORAZIONE richiede nota obbligatoria.")
                return
            if esito_new == "KO" and len(nota_txt) < 5:
                st.error("KO richiede almeno 5 caratteri di nota.")
                return

            try:
                if esito_new == "IN_LAVORAZIONE":
                    event_id = take_in_charge(att_id, nota_txt)
                elif esito_new in ("OK", "KO"):
                    event_id = close_app(att_id, esito_new, nota_txt if nota_txt else None)
                elif esito_new in ("BO_OK", "BO_KO"):
                    event_id = close_bo(att_id, esito_new, nota_txt if nota_txt else None)
                else:
                    st.error("Esito non valido.")
                    return

                riepilogo = st.session_state.get("selected_pratica") or {}
                st.success(
                    "Esito applicato"
                    f"\n\n- event_id: {event_id}"
                    f"\n- attivazione_id: {att_id}"
                    f"\n- esito: {esito_new}"
                    f"\n- nota: {nota_txt if nota_txt else '(nessuna)'}"
                    + (
                        f"\n- telefono: {riepilogo.get('telefono')}"
                        f"\n- seriale_sim: {riepilogo.get('seriale_sim')}"
                        f"\n- pista: {riepilogo.get('pista')}"
                        f"\n- negozio: {riepilogo.get('punto_vendita')}"
                        if riepilogo else ""
                    )
                )

                st.session_state["drill_sel_reset_token"] += 1
                st.session_state.pop("selected_pratica", None)
                st.cache_data.clear()
                st.rerun()

            except Exception as e:
                st.error(f"Errore applicazione esito: {e}")


# ==========================
# NEGOZIO (worklist + KPI tab)
# ==========================
def render_kpi_per_tab(pista_exact: str | None, energia_mode: bool = False) -> None:
    st.subheader("KPI (tab)")

    if energia_mode:
        st.info("Energia: non disponibile in questo dataset.")
        return

    if not pista_exact:
        st.info("KPI non disponibili.")
        return

    kpi = q(
        """
        SELECT
          CASE
            WHEN s.esito_corrente IS NULL THEN 'DA_ESITARE'
            ELSE s.esito_corrente::text
          END AS stato,
          COUNT(*) AS totale
        FROM app.v_attivazione_stato_sec s
        WHERE s.pista = %s
        GROUP BY 1
        ORDER BY
          CASE
            WHEN (CASE WHEN s.esito_corrente IS NULL THEN 'DA_ESITARE' ELSE s.esito_corrente::text END) = 'DA_ESITARE' THEN 0
            WHEN (CASE WHEN s.esito_corrente IS NULL THEN 'DA_ESITARE' ELSE s.esito_corrente::text END) = 'IN_LAVORAZIONE' THEN 1
            WHEN (CASE WHEN s.esito_corrente IS NULL THEN 'DA_ESITARE' ELSE s.esito_corrente::text END) = 'OK' THEN 2
            WHEN (CASE WHEN s.esito_corrente IS NULL THEN 'DA_ESITARE' ELSE s.esito_corrente::text END) = 'KO' THEN 3
            WHEN (CASE WHEN s.esito_corrente IS NULL THEN 'DA_ESITARE' ELSE s.esito_corrente::text END) = 'BO_OK' THEN 4
            WHEN (CASE WHEN s.esito_corrente IS NULL THEN 'DA_ESITARE' ELSE s.esito_corrente::text END) = 'BO_KO' THEN 5
            ELSE 99
          END,
          totale DESC;
        """,
        (pista_exact,),
    )

    if not kpi:
        st.info("Nessun dato per questa tab.")
        return

    cols = st.columns(6)
    for i, row in enumerate(kpi):
        cols[i % len(cols)].metric(label=str(row["stato"]), value=int(row["totale"]))



def render_pratiche_lavorate(pista_exact: str | None, energia_mode: bool = False) -> None:
    st.subheader("Pratiche Lavorate")

    if energia_mode:
        st.info("Energia: non disponibile in questo dataset.")
        return

    if not pista_exact:
        st.info("Nessuna pratica lavorata disponibile.")
        return

    worked_rows = q(
        """
        WITH last_note AS (
          SELECT DISTINCT ON (attivazione_id)
            attivazione_id,
            note
          FROM app.v_esiti_storico_sec
          WHERE note IS NOT NULL AND btrim(note) <> ''
          ORDER BY attivazione_id, created_at DESC
        )
        SELECT
          s.data_attivazione,
          s.telefono,
          s.seriale_sim,
          s.pista,
          s.esito_corrente,
          s.esito_at,
          ln.note AS nota
        FROM app.v_attivazione_stato_sec s
        LEFT JOIN last_note ln USING (attivazione_id)
        WHERE s.esito_corrente IS NOT NULL
          AND s.pista = %s
        ORDER BY s.esito_at DESC NULLS LAST
        LIMIT 300;
        """,
        (pista_exact,),
    )

    if not worked_rows:
        st.info("Nessuna pratica lavorata disponibile.")
        return

    df_worked = pd.DataFrame(worked_rows)
    if "data_attivazione" in df_worked.columns:
        df_worked["data_attivazione"] = pd.to_datetime(df_worked["data_attivazione"], errors="coerce").dt.strftime("%d/%m/%Y")

    st.dataframe(
        df_worked[["data_attivazione", "telefono", "seriale_sim", "pista", "esito_corrente", "nota"]],
        width="stretch",
        hide_index=True,
        column_config={
            "data_attivazione": st.column_config.TextColumn("Data"),
            "telefono": st.column_config.TextColumn("Telefono"),
            "seriale_sim": st.column_config.TextColumn("Seriale SIM"),
            "pista": st.column_config.TextColumn("Pista"),
            "esito_corrente": st.column_config.TextColumn("Esito"),
            "nota": st.column_config.TextColumn("Nota"),
        },
        height=320,
    )



def render_worklist(pista_exact: str | None, energia_mode: bool = False, pista_code: str | None = None) -> None:
    st.subheader("Worklist")

    if energia_mode:
        st.info("Energia: non disponibile in questo dataset.")
        return

    if pista_exact:
        if pista_code == "FAMILY":
            order_clause = "ORDER BY wl.data_attivazione ASC NULLS LAST, wl.sort_at ASC NULLS LAST"
        else:
            order_clause = "ORDER BY wl.sort_at ASC NULLS LAST"

        work_rows = q(
            f"""
            WITH last_note AS (
              SELECT DISTINCT ON (attivazione_id)
                attivazione_id,
                note
              FROM app.v_esiti_storico_sec
              WHERE note IS NOT NULL
                AND btrim(note) <> ''
              ORDER BY attivazione_id, created_at DESC
            )
            SELECT
              wl.attivazione_id,
              wl.data_attivazione,
              wl.telefono,
              wl.seriale_sim,
              wl.pista,
              wl.esito_corrente,
              CASE
                WHEN wl.esito_corrente = 'IN_LAVORAZIONE' THEN ln.note
                ELSE NULL
              END AS nota,
              wl.next_action_label
            FROM app.v_pratiche_lista_sec wl
            LEFT JOIN last_note ln
              ON ln.attivazione_id = wl.attivazione_id
            WHERE wl.next_action_code <> 'SOLO_LETTURA'
              AND wl.pista = %s
            {order_clause}
            LIMIT 500;
            """,
            (pista_exact,),
        )
    else:
        work_rows = []

    if not work_rows:
        st.info("Nessuna pratica lavorabile disponibile.")
        return

    df = pd.DataFrame(work_rows).set_index("attivazione_id")
    df["esito_operatore"] = ""
    df["nota_operatore"] = ""

    visible_cols = [
        "data_attivazione",
        "telefono",
        "seriale_sim",
        "pista",
        "esito_corrente",
        "nota",
        "next_action_label",
    ]

    col_config = {
        "data_attivazione": st.column_config.DateColumn("Data", format="DD/MM/YYYY"),
        "telefono": st.column_config.TextColumn("Telefono"),
        "seriale_sim": st.column_config.TextColumn("Seriale SIM"),
        "pista": st.column_config.TextColumn("Pista"),
        "esito_corrente": st.column_config.TextColumn("Esito attuale"),
        "nota": st.column_config.TextColumn(
            "Nota (attuale)",
            help="Visibile solo se la pratica è IN_LAVORAZIONE (ultima nota non vuota sugli eventi).",
            max_chars=500,
        ),
        "next_action_label": st.column_config.TextColumn("Guida (DB)"),
        "esito_operatore": st.column_config.SelectboxColumn(
            "Esito (nuovo)",
            options=["", "IN_LAVORAZIONE", "OK", "KO"],
            help="IN_LAVORAZIONE: nota obbligatoria • OK: nota facoltativa • KO: nota min 5 caratteri.",
            required=False,
        ),
        "nota_operatore": st.column_config.TextColumn(
            "Nota",
            help="Obbligatoria solo per IN_LAVORAZIONE. Per KO minimo 5 caratteri.",
            max_chars=500,
        ),
    }

    display_cols = visible_cols + ["esito_operatore", "nota_operatore"]
    edited = st.data_editor(
        df[display_cols],
        column_config=col_config,
        hide_index=True,
        width="stretch",
        height=540,
        disabled=visible_cols,
    )

    st.caption("Compila 'Esito (nuovo)' e 'Nota' sulla riga, poi premi **Applica**.")

    if st.button("Applica", type="primary", key=f"apply_{pista_code or pista_exact or 'energia'}"):
        to_apply = edited[edited["esito_operatore"].astype(str).str.strip() != ""].copy()
        if to_apply.empty:
            st.info("Nessuna modifica da applicare.")
            return

        errors: list[str] = []
        applied = 0

        for att_id, row in to_apply.iterrows():
            att_id = int(att_id)
            esito_new = str(row.get("esito_operatore") or "").strip()
            nota = str(row.get("nota_operatore") or "").strip()

            if esito_new == "IN_LAVORAZIONE":
                if not nota:
                    errors.append(f"{att_id}: IN_LAVORAZIONE richiede nota.")
                    continue
                try:
                    take_in_charge(att_id, nota)
                    applied += 1
                except Exception as e:
                    errors.append(f"{att_id}: errore IN_LAVORAZIONE -> {e}")

            elif esito_new in ("OK", "KO"):
                if esito_new == "KO" and len(nota) < 5:
                    errors.append(f"{att_id}: KO richiede almeno 5 caratteri di nota.")
                    continue
                try:
                    close_app(att_id, esito_new, nota if nota else None)
                    applied += 1
                except Exception as e:
                    errors.append(f"{att_id}: errore {esito_new} -> {e}")

            else:
                errors.append(f"{att_id}: esito non valido.")

        if applied:
            st.success(f"Operazioni applicate: {applied}")
            st.cache_data.clear()
            st.rerun()

        if errors:
            st.error("Alcune righe non sono state applicate:")
            for msg in errors:
                st.write("- " + msg)


# ==========================
# ROUTING UI
# ==========================
if dashboard_mode:
    render_dashboard_esiti()
else:
    st.subheader("Worklist • operativa")
    tab_mnp, tab_family, tab_energia = st.tabs(["MNP", "Family", "Energia"])

    with tab_mnp:
        pista = pista_value("MNP")
        render_kpi_per_tab(pista)
        st.divider()
        render_pratiche_lavorate(pista)
        st.divider()
        render_worklist(pista, pista_code="MNP")

    with tab_family:
        pista = pista_value("FAMILY")
        render_kpi_per_tab(pista)
        st.divider()
        render_pratiche_lavorate(pista)
        st.divider()
        render_worklist(pista, pista_code="FAMILY")

    with tab_energia:
        render_kpi_per_tab(None, energia_mode=True)
        st.divider()
        render_pratiche_lavorate(None, energia_mode=True)
        st.divider()
        render_worklist(None, energia_mode=True)

st.divider()
st.caption(f"Ruolo DB effettivo: {db_role}")
