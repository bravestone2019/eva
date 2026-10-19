"""Small SQLite account and conversation store for a single EVA server."""

import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
import time
import uuid
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
YAHOO_DOMAIN_PATTERN = re.compile(r"yahoo\.[a-z]{2,3}(?:\.[a-z]{2})?\Z")
DB_PATH = Path(os.getenv("EVA_ACCOUNT_DB", "data/eva_accounts.sqlite3"))
if not DB_PATH.is_absolute():
    DB_PATH = PROJECT_ROOT / DB_PATH
TOKEN_SECRET = os.getenv("EVA_AUTH_SECRET", "").encode("utf-8")
if not TOKEN_SECRET:
    # Stable on this server across restarts; production deployments should set EVA_AUTH_SECRET.
    secret_path = DB_PATH.with_suffix(".secret")
    secret_path.parent.mkdir(parents=True, exist_ok=True)
    if secret_path.exists():
        TOKEN_SECRET = secret_path.read_bytes()
    else:
        TOKEN_SECRET = secrets.token_bytes(32)
        secret_path.write_bytes(TOKEN_SECRET)


def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, timeout=15)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode=WAL")
    return connection


def initialize():
    with _connect() as db:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            email TEXT NOT NULL UNIQUE,
            salt BLOB NOT NULL,
            password_hash BLOB NOT NULL,
            created_at INTEGER NOT NULL,
            google_sub TEXT
        );
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT NOT NULL,
            user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            title TEXT NOT NULL,
            updated_at INTEGER NOT NULL,
            messages_json TEXT NOT NULL,
            PRIMARY KEY (user_id, id)
        );
        """)
        columns = {row["name"] for row in db.execute("PRAGMA table_info(users)")}
        if "google_sub" not in columns:
            db.execute("ALTER TABLE users ADD COLUMN google_sub TEXT")
        db.execute("CREATE UNIQUE INDEX IF NOT EXISTS users_google_sub_unique ON users(google_sub)")


def validate_email(email):
    email = str(email).strip().lower()
    if len(email) > 254 or email.count("@") != 1:
        raise ValueError("Enter a valid Gmail or Yahoo email address.")
    local, domain = email.rsplit("@", 1)
    if (
        not local
        or len(local) > 64
        or local.startswith(".")
        or local.endswith(".")
        or ".." in local
        or not re.fullmatch(r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+", local)
    ):
        raise ValueError("Enter a valid Gmail or Yahoo email address.")
    if domain != "gmail.com" and not YAHOO_DOMAIN_PATTERN.fullmatch(domain):
        raise ValueError("Use a Gmail or Yahoo email address.")
    return email


def _password_hash(password, salt):
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 310_000)


def create_user(email, password):
    email = validate_email(email)
    if not re.fullmatch(r"\d{6}", str(password)):
        raise ValueError("Password must contain exactly 6 digits.")
    user_id, salt = uuid.uuid4().hex, secrets.token_bytes(16)
    try:
        with _connect() as db:
            db.execute("INSERT INTO users (id,email,salt,password_hash,created_at) VALUES (?, ?, ?, ?, ?)",
                       (user_id, email, salt, _password_hash(password, salt), int(time.time())))
    except sqlite3.IntegrityError as error:
        raise ValueError("An account with that email already exists.") from error
    return {"id": user_id, "email": email}


def authenticate(email, password):
    try:
        email = validate_email(email)
    except ValueError:
        return None
    with _connect() as db:
        row = db.execute("SELECT id,email,salt,password_hash FROM users WHERE email=?", (email,)).fetchone()
    if not row or not hmac.compare_digest(row["password_hash"], _password_hash(password, row["salt"])):
        return None
    return {"id": row["id"], "email": row["email"]}


def google_user(email, google_sub):
    """Find or create an account using an already-verified Google identity."""
    email = validate_email(email)
    google_sub = str(google_sub).strip()
    if not google_sub or len(google_sub) > 255:
        raise ValueError("Google did not return a valid account identifier.")
    with _connect() as db:
        row = db.execute("SELECT id,email FROM users WHERE google_sub=?", (google_sub,)).fetchone()
        if row:
            if row["email"].lower() != email:
                try:
                    db.execute("UPDATE users SET email=? WHERE id=?", (email, row["id"]))
                except sqlite3.IntegrityError as error:
                    raise ValueError("This verified email is already used by another EVA account.") from error
                return {"id": row["id"], "email": email}
            return {"id": row["id"], "email": row["email"]}

        row = db.execute("SELECT id,email,google_sub FROM users WHERE email=?", (email,)).fetchone()
        if row:
            if row["google_sub"] and row["google_sub"] != google_sub:
                raise ValueError("This email is already linked to a different Google account.")
            db.execute("UPDATE users SET google_sub=? WHERE id=?", (google_sub, row["id"]))
            return {"id": row["id"], "email": row["email"]}

        user_id = uuid.uuid4().hex
        salt = secrets.token_bytes(16)
        random_password = secrets.token_urlsafe(48)
        db.execute(
            "INSERT INTO users (id,email,salt,password_hash,created_at,google_sub) VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, email, salt, _password_hash(random_password, salt), int(time.time()), google_sub),
        )
        return {"id": user_id, "email": email}


def issue_token(user_id):
    expires = int(time.time()) + 60 * 60 * 24 * 30
    payload = f"{user_id}.{expires}"
    signature = hmac.new(TOKEN_SECRET, payload.encode(), hashlib.sha256).hexdigest()
    return f"{payload}.{signature}"


def verify_token(token):
    try:
        user_id, expires, signature = token.split(".")
        payload = f"{user_id}.{expires}"
        expected = hmac.new(TOKEN_SECRET, payload.encode(), hashlib.sha256).hexdigest()
        if int(expires) < time.time() or not hmac.compare_digest(signature, expected):
            return None
        with _connect() as db:
            row = db.execute("SELECT id,email FROM users WHERE id=?", (user_id,)).fetchone()
        return dict(row) if row else None
    except (ValueError, TypeError):
        return None


def list_conversations(user_id):
    with _connect() as db:
        rows = db.execute("SELECT id,title,updated_at,messages_json FROM conversations WHERE user_id=? ORDER BY updated_at DESC", (user_id,)).fetchall()
    return [{"id": r["id"], "title": r["title"], "updatedAt": r["updated_at"], "messages": json.loads(r["messages_json"])} for r in rows]


def save_conversation(user_id, value):
    conversation_id = str(value.get("id", ""))
    title = str(value.get("title", "New conversation"))[:120]
    messages = value.get("messages", [])
    if not conversation_id or len(conversation_id) > 80 or not isinstance(messages, list) or len(messages) > 5000:
        raise ValueError("Invalid conversation data.")
    updated = int(value.get("updatedAt", time.time() * 1000))
    payload = json.dumps(messages, ensure_ascii=False)
    with _connect() as db:
        db.execute("""INSERT INTO conversations VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(user_id,id) DO UPDATE SET title=excluded.title,updated_at=excluded.updated_at,messages_json=excluded.messages_json""",
            (conversation_id, user_id, title, updated, payload))


def delete_conversation(user_id, conversation_id):
    with _connect() as db:
        result = db.execute("DELETE FROM conversations WHERE user_id=? AND id=?", (user_id, conversation_id))
    return result.rowcount > 0


initialize()
