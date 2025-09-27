# Import key components to make them accessible from the auth package
# Import directly from models.py to avoid circular import
from app.auth.models import User
from app.auth.manager import current_active_user, current_superuser