"""Container startup/maintenance boundaries with no model or real database."""
import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from sqlalchemy.engine import make_url

ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = ROOT / 'docker/backend-entrypoint.py'
# The image keeps its startup script outside source bind mounts.
if not ENTRYPOINT.is_file():
    ENTRYPOINT = Path('/usr/local/bin/livestream-backend.py')
spec = importlib.util.spec_from_file_location('container_entrypoint', ENTRYPOINT)
entrypoint = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entrypoint)


class ContainerEntrypointTests(unittest.TestCase):
    def credentials(self):
        return {'POSTGRES_DB': 'review', 'POSTGRES_USER': 'review_user',
                'POSTGRES_PASSWORD': 'p@ss:word/#% with spaces', 'DB_HOST': 'postgres', 'DB_PORT': '5432',
                'DATABASE_URL': 'sqlite:///host-only.db'}

    def test_container_url_roundtrips_special_password_and_ignores_native_url(self):
        values = self.credentials()
        url = make_url(entrypoint.database_url(values))
        self.assertEqual(url.password, values['POSTGRES_PASSWORD'])
        self.assertEqual((url.drivername, url.host, url.port, url.database),
                         ('postgresql+psycopg', 'postgres', 5432, 'review'))

    def test_missing_or_template_credentials_fail_without_echoing_password(self):
        for values in ({}, dict(self.credentials(), POSTGRES_PASSWORD='REPLACE_WITH_A_RANDOM_PASSWORD')):
            with self.subTest(values_present=bool(values)), self.assertRaises(ValueError) as caught:
                entrypoint.database_url(values)
            self.assertNotIn('p@ss', str(caught.exception))

    def test_explicit_test_command_does_not_migrate_or_start_models(self):
        with patch.dict(os.environ, self.credentials(), clear=True), \
             patch.object(entrypoint.sys, 'argv', ['entrypoint', 'python', '-m', 'unittest']), \
             patch.object(entrypoint.os, 'execvp') as execute, \
             patch.object(entrypoint.subprocess, 'run') as migrate:
            entrypoint.main()
            execute.assert_called_once_with('python', ['python', '-m', 'unittest'])
            migrate.assert_not_called()
            self.assertEqual(make_url(os.environ['DATABASE_URL']).host, 'postgres')

    def test_postgres_diagnostics_use_container_address_and_native_default(self):
        from backend.database_transfer import postgres_url
        values = self.credentials()
        with patch('backend.database_transfer.load_dotenv'), patch.dict(os.environ, values, clear=True):
            self.assertEqual((postgres_url().host, postgres_url().port), ('postgres', 5432))
        del values['DB_HOST']; del values['DB_PORT']
        with patch('backend.database_transfer.load_dotenv'), patch.dict(os.environ, values, clear=True):
            self.assertEqual((postgres_url().host, postgres_url().port), ('127.0.0.1', 5433))
