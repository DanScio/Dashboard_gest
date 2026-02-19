import os
from datetime import datetime
from pathlib import Path

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
# ENV (.env + DOTENV_PATH)
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

with st.expander("Debug config (.env)"):
    st.write("DOTENV_PATH:", os.getenv("DOTENV_PATH"))
    st.write("DB_HOST:", os.getenv("DB_HOST"))
    st.write("DB_PORT:", os.getenv("DB_PORT"))
    st.write("DB_NAME:", os.getenv("DB_NAME"))
    st.write("DB_USER:", os.getenv("DB_USER"))
    st.write("DB_PASSWORD presente:", bool(os.getenv("DB_PASSWORD")))
    st.write("DB_DSN/DATABASE_URL presente:", bool(os.getenv("DB_DSN") or os.getenv("DATABASE_URL")))
    st.write("DB_ROLE (effettivo):", os.getenv("DB_ROLE") or "app_readonly")


def get_db_dsn() -> str:
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
    return os.getenv("DB_ROLE") or "app_readonly"


def is_store_role(db_role: str) -> bool:
    return (db_role or "").startswith("op_")


def is_dashboard_role(db_role: str) -> bool:
    return db_role in ("app_owner", "dashboard_owner")


# -------------------------
# DB (cache + SET ROLE immediato)
# -------------------------
@st.cache_data(ttl=15)
def query(sql: str, params: tuple | None, db_role: str) -> list[dict]:
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
# UI
# -------------------------
db_role = get_db_role()
store_mode = is_store_role(db_role)
dashboard_mode = is_dashboard_role(db_role)

st.title("Dashboard Gest")
st.caption(f"Aggiornato: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")

if st.button("Aggiorna dati"):
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
    st.warning("Ruolo DB non riconosciuto per questa UI. Controlla DB_ROLE nel .env.")
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
    """
    Ritorna (selected_state, selected_pista) SOLO se l'utente clicca una fetta.
    """
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
    pista_value = pista_value(pista_code) if pista_code is not None else None


    # KPI (torte) + selezione stato SOLO via click
    selected_state = None
    selected_pista_for_drill = None

    if pista_value is None:
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
        s, psel = render_one_pie(pv_id, pista_value, "Distribuzione esiti", "single")
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

    # token per resettare SEL (auto-reset dopo apply)
    if "drill_sel_reset_token" not in st.session_state:
        st.session_state["drill_sel_reset_token"] = 0

    selected_att_id = None

    if get_db_role() == "app_owner":
        # 1) nascondo attivazione_id (lo metto come index)
        df_edit = df_drill.copy().set_index("attivazione_id")
        df_edit.insert(0, "SEL", False)

        edited = st.data_editor(
            df_edit,
            key=f"drill_editor_{st.session_state['drill_sel_reset_token']}",
            hide_index=True,  # attivazione_id resta nascosto
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

            # 2) highlight "soft": card riepilogo (riga selezionata)
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

            # salvo dettagli per conferma (punto 4) e per default nota (punto 5)
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

    # --------------------------
    # GESTIONE ESITI (SOLO app_owner)
    # --------------------------
    if get_db_role() == "app_owner":
        st.divider()
        st.subheader("Gestione esiti (app_owner)")

        pratica = st.session_state.get("selected_pratica") or {}
        default_id = int(pratica.get("attivazione_id") or 1)

        # 5) extra: precompilo nota con nota attuale (se presente) + mostra contesto
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

            # Validazioni: identiche allo Store mode
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

                # 4) conferma con riepilogo (pratica + azione)
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

                # 3) auto-reset SEL: cambia key del data_editor + pulisci selezione
                st.session_state["drill_sel_reset_token"] += 1
                st.session_state.pop("selected_pratica", None)

                st.cache_data.clear()
                st.rerun()

            except Exception as e:
                st.error(f"Errore applicazione esito: {e}")


# ==========================
# NEGOZIO (worklist + KPI tab) - invariato dove non richiesto
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


def render_worklist(pista_exact: str | None, energia_mode: bool = False) -> None:
    st.subheader("Worklist")

    if energia_mode:
        st.info("Energia: non disponibile in questo dataset.")
        return

    if pista_exact:
        # Ordering: MNP resta invariato; FAMILY forzato "dal più vecchio al più nuovo"
        if pista_exact == "FAMILY":
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

    if st.button("Applica", type="primary", key=f"apply_{pista_exact or 'energia'}"):
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
        render_worklist(pista, pista_code= "FAMILY")

    with tab_energia:
        render_kpi_per_tab(None, energia_mode=True)
        st.divider()
        render_pratiche_lavorate(None, energia_mode=True)
        st.divider()
        render_worklist(None, energia_mode=True)

st.divider()
st.caption(f"Ruolo DB effettivo: {db_role}")
