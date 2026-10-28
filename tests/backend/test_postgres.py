"""Opt-in integration tests against fresh disposable PostgreSQL databases only."""
import os
import unittest
from uuid import uuid4
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.engine import make_url
from backend.database import Database
from backend.database_transfer import postgres_url, snapshot, copy_rows, verify_rows
from tests.backend import test_history as history_tests
from tests.backend import test_live as live_tests
from tests.backend import test_review_data as review_tests


@unittest.skipUnless(os.environ.get('RUN_POSTGRES_TESTS') == '1', 'Set RUN_POSTGRES_TESTS=1 to test PostgreSQL')
class PostgresHistoryTests(history_tests.HistoryTests):
    def setUp(self):
        url = postgres_url()
        self.test_name = 'livestream_test_' + uuid4().hex
        self.admin_engine = create_engine(url, isolation_level='AUTOCOMMIT')
        with self.admin_engine.connect() as connection:
            connection.execute(text(f'CREATE DATABASE "{self.test_name}"'))
        self.test_url = url.set(database=self.test_name)
        super().setUp()

    def database_url(self):
        return self.test_url

    def tearDown(self):
        try:
            super().tearDown()
        finally:
            # Name constructed in this test only; never a user-supplied/main database name.
            if not self.test_name.startswith('livestream_test_') or len(self.test_name) != 48:
                raise RuntimeError('Refusing to drop an unexpected database')
            with self.admin_engine.connect() as connection:
                connection.execute(text(f'DROP DATABASE "{self.test_name}"'))
            self.admin_engine.dispose()

    def test_schema_and_import_idempotency_conflict_rollback(self):
        self.upload()
        database = self.app.state.database
        with database.engine.connect() as connection:
            rows = snapshot(connection)
            inspector = inspect(connection)
            self.assertEqual(len(inspector.get_foreign_keys('analysis_segments')), 1)
            self.assertEqual(len(inspector.get_foreign_keys('feedback')), 1)
            self.assertGreaterEqual(len(inspector.get_indexes('analysis_segments')), 2)
            self.assertGreaterEqual(len(inspector.get_check_constraints('analysis_segments')), 2)
        with database.engine.begin() as connection:
            self.assertTrue(all(count == 0 for count in copy_rows(rows, connection).values()))
            self.assertTrue(verify_rows(rows, snapshot(connection))['all_source_rows_equal'])
        import copy
        modified = copy.deepcopy(rows)
        extra = dict(modified['analysis_sessions'][0], id=str(uuid4()))
        modified['analysis_sessions'].append(extra)
        modified['analysis_segments'][0]['text'] = 'Different destination contents'
        with self.assertRaises(ValueError):
            with database.engine.begin() as connection:
                copy_rows(modified, connection)
        with database.engine.connect() as connection:
            self.assertEqual(snapshot(connection), rows)


@unittest.skipUnless(os.environ.get('RUN_POSTGRES_TESTS') == '1', 'Set RUN_POSTGRES_TESTS=1')
class PostgresLiveTests(live_tests.LiveTests):
    def setUp(self):
        self.test_name='livestream_test_'+uuid4().hex
        self.admin_engine=create_engine(postgres_url(),isolation_level='AUTOCOMMIT')
        with self.admin_engine.connect() as connection:
            connection.execute(text(f'CREATE DATABASE "{self.test_name}"'))
        self.test_url=postgres_url().set(database=self.test_name)
        super().setUp()

    def database_url(self):
        return self.test_url

    def tearDown(self):
        try:
            super().tearDown()
        finally:
            if not self.test_name.startswith('livestream_test_') or len(self.test_name)!=48:
                raise RuntimeError('Refusing to drop unexpected database')
            with self.admin_engine.connect() as connection:
                connection.execute(text(f'DROP DATABASE "{self.test_name}"'))
            self.admin_engine.dispose()


@unittest.skipUnless(os.environ.get('RUN_POSTGRES_TESTS') == '1', 'Set RUN_POSTGRES_TESTS=1')
class PostgresReviewTests(review_tests.ReviewTests):
    def setUp(self):
        self.test_name='livestream_test_'+uuid4().hex
        self.admin_engine=create_engine(postgres_url(),isolation_level='AUTOCOMMIT')
        with self.admin_engine.connect() as connection:
            connection.execute(text(f'CREATE DATABASE "{self.test_name}"'))
        self.test_url=postgres_url().set(database=self.test_name)
        super().setUp()

    def database_url(self):return self.test_url

    def tearDown(self):
        try:super().tearDown()
        finally:
            if not self.test_name.startswith('livestream_test_') or len(self.test_name)!=48:
                raise RuntimeError('Refusing to drop unexpected database')
            with self.admin_engine.connect() as connection:
                connection.execute(text(f'DROP DATABASE "{self.test_name}"'))
            self.admin_engine.dispose()
