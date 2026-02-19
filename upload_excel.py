# upload_excel.py
# Wrapper "1 comando" che:
# 1) prepara .env (opzionale) leggendo DOTENV_PATH (come fai in main.py)
# 2) esegue import/import_excel_to_staging.py per uno o più fogli
# 3) esegue db/load_from_staging.sql passando il batch_id
#
# NON modifica import_excel_to_staging.py e NON modifica load_from_staging.sql

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


BATCH_RE = re.compile(r"DEBUG:\s*batch_id\s*import\s*=\s*([0-9a-fA-F-]{36})")


def project_root() -> Path:
    return Path(__file__).resolve().parent


def resolve_dotenv_path(root: Path) -> Path | None:
    dp = os.getenv("DOTENV_PATH")
    if not dp:
        return None
    p = Path(dp)
    if not p.is_absolute():
        p = root / p
    return p


def ensure_env_from_dotenv_path(root: Path) -> tuple[bool, Path | None, Path | None]:
    """
    Se DOTENV_PATH è settato e punta a un file esistente, copia quel file in .env (root),
    facendo backup dell'eventuale .env esistente.
    Ritorna:
      (did_copy, backup_path, dotenv_path_used)
    """
    dotenv_used = resolve_dotenv_path(root)
    if not dotenv_used:
        return (False, None, None)

    if not dotenv_used.exists():
        raise FileNotFoundError(f"DOTENV_PATH punta a un file inesistente: {dotenv_used}")

    target_env = root / ".env"
    backup = None

    if target_env.exists():
        backup = root / ".env.bak_upload_excel"
        shutil.copyfile(target_env, backup)

    shutil.copyfile(dotenv_used, target_env)
    return (True, backup, dotenv_used)


def restore_env(root: Path, did_copy: bool, backup: Path | None) -> None:
    if not did_copy:
        return

    target_env = root / ".env"
    if backup and backup.exists():
        shutil.copyfile(backup, target_env)
        backup.unlink(missing_ok=True)
    else:
        # Se non c'era un .env prima, lo rimuoviamo (per non lasciare tracce)
        target_env.unlink(missing_ok=True)


def run_import(root: Path, excel_path: Path, sheet_name: str) -> str:
    """
    Esegue import/import_excel_to_staging.py e ritorna il batch_id.
    """
    script = root / "import" / "import_excel_to_staging.py"
    if not script.exists():
        raise FileNotFoundError(f"Script import non trovato: {script}")

    cmd = [sys.executable, str(script), str(excel_path), sheet_name]
    proc = subprocess.run(
        cmd,
        cwd=str(root),
        capture_output=True,
        text=True,
    )

    out = (proc.stdout or "") + "\n" + (proc.stderr or "")

    if proc.returncode != 0:
        raise RuntimeError(f"Import fallito (foglio '{sheet_name}'). Output:\n{out}")

    m = BATCH_RE.search(out)
    if not m:
        raise RuntimeError(
            f"Import ok ma batch_id non trovato nell'output (foglio '{sheet_name}'). Output:\n{out}"
        )

    return m.group(1)


def run_load_psql(root: Path, batch_id: str, dbname: str, dbuser: str | None) -> None:
    """
    Esegue db/load_from_staging.sql passando -v batch_id=...
    """
    sql_file = root / "db" / "load_from_staging.sql"
    if not sql_file.exists():
        raise FileNotFoundError(f"SQL loader non trovato: {sql_file}")

    cmd = ["psql", "-d", dbname]
    if dbuser:
        cmd += ["-U", dbuser]
    cmd += ["-v", f"batch_id={batch_id}", "-f", str(sql_file)]

    proc = subprocess.run(
        cmd,
        cwd=str(root),
        capture_output=True,
        text=True,
    )

    if proc.returncode != 0:
        out = (proc.stdout or "") + "\n" + (proc.stderr or "")
        raise RuntimeError(f"Load psql fallito (batch_id={batch_id}). Output:\n{out}")


def main() -> int:
    root = project_root()

    ap = argparse.ArgumentParser(
        description="Upload Excel -> staging -> load in app.attivazione (1 comando)."
    )
    ap.add_argument("excel_path", help="Path del file Excel (.xlsx)")
    ap.add_argument("--suite", help="Nome foglio per MNP/FAMILY (es. 'DE SUITE')", default=None)
    ap.add_argument("--energia", help="Nome foglio per ENERGIA (es. 'DE ENERGIA')", default=None)
    ap.add_argument(
        "--db",
        help="Nome database (default: dashboard_gest)",
        default="dashboard_gest",
    )
    ap.add_argument(
        "--psql-user",
        help="Utente psql per eseguire il load (default: app_owner). Se non vuoi passarlo, ometti.",
        default="app_owner",
    )
    ap.add_argument(
        "--no-dotenv-copy",
        action="store_true",
        help="Non copiare DOTENV_PATH in .env (usa .env già presente).",
    )

    args = ap.parse_args()

    excel = Path(args.excel_path).resolve()
    if not excel.exists():
        print(f"ERRORE: file Excel non trovato: {excel}")
        return 2

    sheets: list[tuple[str, str]] = []
    if args.suite:
        sheets.append(("SUITE", args.suite))
    if args.energia:
        sheets.append(("ENERGIA", args.energia))

    if not sheets:
        print("ERRORE: devi specificare almeno --suite oppure --energia")
        return 2

    did_copy = False
    backup = None
    dotenv_used = None

    try:
        if not args.no_dotenv_copy:
            did_copy, backup, dotenv_used = ensure_env_from_dotenv_path(root)

        if did_copy:
            print(f"OK: .env preparato da DOTENV_PATH -> {dotenv_used}")

        results: list[tuple[str, str]] = []  # (label, batch_id)

        for label, sheet in sheets:
            print(f"\n== IMPORT {label} | foglio: {sheet} ==")
            batch_id = run_import(root, excel, sheet)
            print(f"OK: batch_id {label} = {batch_id}")
            results.append((label, batch_id))

        for label, batch_id in results:
            print(f"\n== LOAD {label} | batch_id: {batch_id} ==")
            run_load_psql(root, batch_id, dbname=args.db, dbuser=args.psql_user)
            print(f"OK: load completato per {label}")

        print("\nTUTTO OK.")
        return 0

    except Exception as e:
        print(f"\nERRORE: {e}")
        return 1

    finally:
        try:
            restore_env(root, did_copy, backup)
        except Exception:
            # non blocchiamo il processo per problemi di restore
            pass


if __name__ == "__main__":
    raise SystemExit(main())

