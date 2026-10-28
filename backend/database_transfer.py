"""Consistent SQLite backup and transactional, conflict-safe PostgreSQL import.

Run with python -m backend.database_transfer. Never prints connection strings.
"""
import argparse
import hashlib
import json
import os
import sqlite3
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, select, text, inspect
from sqlalchemy.engine import URL
from backend.database import Base, Database, ROOT

TABLES = [Base.metadata.tables[name] for name in ('analysis_sessions', 'analysis_segments', 'feedback')]


def postgres_url():
    load_dotenv(ROOT / '.env', override=False)
    return URL.create('postgresql+psycopg', username=os.environ['POSTGRES_USER'],
        password=os.environ['POSTGRES_PASSWORD'], host=os.environ.get('DB_HOST', '127.0.0.1'),
        port=int(os.environ.get('DB_PORT', os.environ.get('POSTGRES_PORT', '5433'))), database=os.environ['POSTGRES_DB'])


def canonical(rows):
    return json.dumps(sorted(rows, key=lambda row: row['id']), ensure_ascii=False,
                      sort_keys=True, separators=(',', ':'), allow_nan=False)


def snapshot(connection):
    # Historical SQLite backups predate optional corrected-feedback columns.
    # Read only columns actually present; never migrate a read-only backup.
    result = {}
    inspector = inspect(connection)
    for table in TABLES:
        present = {column['name'] for column in inspector.get_columns(table.name)}
        query = select(*(column for column in table.c if column.name in present))
        result[table.name] = [dict(row) for row in connection.execute(query).mappings()]
    return result


def summary(rows):
    return {name: {'count': len(values), 'sha256': hashlib.sha256(canonical(values).encode('utf-8')).hexdigest()}
            for name, values in rows.items()}


def check_relationships(rows):
    sessions = {r['id'] for r in rows['analysis_sessions']}
    segments = {r['id'] for r in rows['analysis_segments']}
    if any(r['session_id'] not in sessions for r in rows['analysis_segments']):
        raise ValueError('Orphan analysis segment')
    if any(r['segment_id'] not in segments for r in rows['feedback']):
        raise ValueError('Orphan feedback')


def backup_sqlite(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if not source.is_file() or source == destination:
        raise ValueError('Source must be an existing SQLite file; destination must be different')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.touch(exist_ok=False)  # never overwrite a previous backup
    with sqlite3.connect(source.as_uri() + '?mode=ro', uri=True) as src, sqlite3.connect(destination) as dst:
        src.backup(dst)
        if dst.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
            raise ValueError('SQLite backup failed integrity check')
    engine = create_engine(URL.create('sqlite', database=str(destination)))
    try:
        with engine.connect() as connection:
            rows = snapshot(connection)
            check_relationships(rows)
            return {'backup': str(destination), 'tables': summary(rows), 'relationships_ok': True}
    finally:
        engine.dispose()


def copy_rows(source_rows, connection):
    """Called within one transaction; identical IDs skipped, conflicts abort all writes."""
    check_relationships(source_rows)
    inserted = {}
    for table in TABLES:
        existing = {row['id']: dict(row) for row in connection.execute(select(table)).mappings()}
        additions = []
        for row in source_rows[table.name]:
            previous = existing.get(row['id'])
            if previous is not None:
                if canonical([{key:previous.get(key) for key in row}]) != canonical([row]):
                    raise ValueError(f'Conflicting destination row: {table.name}/{row["id"]}; nothing overwritten')
            else:
                additions.append(row)
        if additions:
            connection.execute(table.insert(), additions)
        inserted[table.name] = len(additions)
    return inserted


def verify_rows(source_rows, destination_rows):
    check_relationships(destination_rows)
    for table in TABLES:
        target = {row['id']: row for row in destination_rows[table.name]}
        matched = []
        for row in source_rows[table.name]:
            if row['id'] not in target:
                raise ValueError(f'Missing destination row: {table.name}/{row["id"]}')
            matched.append({key:target[row['id']].get(key) for key in row})
        if canonical(matched) != canonical(source_rows[table.name]):
            raise ValueError(f'Content mismatch: {table.name}')
    return {'source': summary(source_rows), 'destination': summary(destination_rows),
            'all_source_rows_equal': True, 'relationships_ok': True}


def transfer(source, target_url, verify_only=False):
    source_engine = create_engine(URL.create('sqlite', database=str(Path(source).resolve())))
    target = Database(target_url)
    try:
        if target.backend != 'postgresql':
            raise ValueError('Import destination must be PostgreSQL')
        with source_engine.connect() as connection:
            source_rows = snapshot(connection)
        target.migrate()
        with target.engine.begin() as connection:
            # Serialize this project's imports and block concurrent writes during comparison.
            connection.execute(text('SELECT pg_advisory_xact_lock(814052)'))
            connection.execute(text('LOCK TABLE analysis_sessions, analysis_segments, feedback IN SHARE ROW EXCLUSIVE MODE'))
            inserted = {} if verify_only else copy_rows(source_rows, connection)
            result = verify_rows(source_rows, snapshot(connection))
            result['inserted'] = inserted
        return result
    finally:
        source_engine.dispose()
        target.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    backup = sub.add_parser('backup-sqlite')
    backup.add_argument('--source', default=str(ROOT / 'runtime' / 'livestream.sqlite3'))
    backup.add_argument('--output', required=True)
    for action in ('import-postgres', 'verify-postgres'):
        item = sub.add_parser(action)
        item.add_argument('--source', required=True, help='Use a consistent SQLite backup')
        item.add_argument('--report', help='Optional JSON result file; contains no credentials')
    args = parser.parse_args()
    try:
        if args.action == 'backup-sqlite':
            result = backup_sqlite(args.source, args.output)
        else:
            if not Path(args.source).is_file():
                raise ValueError('SQLite source not found')
            result = transfer(args.source, postgres_url(), args.action == 'verify-postgres')
            if args.report:
                path = Path(args.report)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, FileExistsError) as error:
        parser.exit(1, f'{error}\n')
    except Exception:
        # Driver errors may contain SQL parameters or credentials. Never print them here.
        parser.exit(1, 'Database operation failed. Check service availability/configuration; no credentials printed.\n')


if __name__ == '__main__':
    main()
