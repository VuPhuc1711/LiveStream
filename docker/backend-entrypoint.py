"""Container startup only; native Windows configuration remains unchanged."""
import os
from pathlib import Path
import subprocess
import sys

from sqlalchemy.engine import URL


def database_url(environ):
    """Build a correctly escaped URL; never print it or inherit host .env URLs."""
    required = ('POSTGRES_DB', 'POSTGRES_USER', 'POSTGRES_PASSWORD')
    if any(not environ.get(name) for name in required):
        raise ValueError('Container needs POSTGRES_DB, POSTGRES_USER and POSTGRES_PASSWORD.')
    if environ['POSTGRES_PASSWORD'] == 'REPLACE_WITH_A_RANDOM_PASSWORD':
        raise ValueError('Set a local PostgreSQL password before starting the container.')
    return URL.create(
        'postgresql+psycopg', username=environ['POSTGRES_USER'],
        password=environ['POSTGRES_PASSWORD'], host=environ.get('DB_HOST', 'postgres'),
        port=int(environ.get('DB_PORT', '5432')), database=environ['POSTGRES_DB'],
    ).render_as_string(hide_password=False)


def main():
    command = sys.argv[1:] or ['serve']
    serving = command == ['serve']
    if serving or all(os.environ.get(name) for name in ('POSTGRES_DB', 'POSTGRES_USER', 'POSTGRES_PASSWORD')):
        os.environ['DATABASE_URL'] = database_url(os.environ)
    if not serving:
        # Explicit maintenance/test commands do not migrate or load either model.
        os.execvp(command[0], command)
        return
    # Host root .env is excluded from the image and never mounted into /app.
    # Credentials remain process environment only; no generated config file.
    runtime = Path('/app/runtime')
    runtime.mkdir(exist_ok=True)
    if not os.access(runtime, os.W_OK):
        raise PermissionError('Container user cannot write /app/runtime; check bind-mount permissions.')
    subprocess.run([sys.executable, '-m', 'tools.check_container_assets'], check=True)
    subprocess.run([sys.executable, '-m', 'alembic', 'upgrade', 'head'], check=True)
    command = [sys.executable, '-X', 'utf8', '-m', 'uvicorn', 'backend.app:app',
               '--host', '0.0.0.0', '--port', '8000', '--workers', '1',
               '--ws-max-size', '65536', '--ws-max-queue', '8',
               '--timeout-graceful-shutdown', '60']
    os.execv(sys.executable, command)


if __name__ == '__main__':
    main()
