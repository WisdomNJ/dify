"""empty message

Revision ID: cd9faf7d567a
Revises: 0c79d303c76d, 582c477e905b, 1c9ba48be8e4
Create Date: 2025-07-17 09:12:27.219210

"""
from alembic import op
import models as models
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'cd9faf7d567a'
down_revision = ('0c79d303c76d', '582c477e905b', '1c9ba48be8e4')
branch_labels = None
depends_on = None


def upgrade():
    pass


def downgrade():
    pass
