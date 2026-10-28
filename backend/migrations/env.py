from alembic import context
from backend.database import Base, Database

config = context.config


def run(connection):
    context.configure(connection=connection, target_metadata=Base.metadata,
                      render_as_batch=connection.dialect.name == 'sqlite')
    with context.begin_transaction():
        context.run_migrations()


if context.is_offline_mode():
    raise RuntimeError('Use online migrations for the configured database.')
elif config.attributes.get('connection') is not None:
    run(config.attributes['connection'])
else:
    database = Database()
    try:
        with database.engine.begin() as connection:
            run(connection)
    finally:
        database.close()
