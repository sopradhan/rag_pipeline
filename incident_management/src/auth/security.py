from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from jose import JWTError, jwt
from passlib.context import CryptContext
from .models import User, Role
from fastapi import HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer

# Security configuration
SECRET_KEY = "your-secret-key"  # Move to environment variables
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class AuthManager:
    def __init__(self, db_session: Session):
        self.db = db_session

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        return pwd_context.verify(plain_password, hashed_password)

    def get_password_hash(self, password: str) -> str:
        """Generate password hash."""
        return pwd_context.hash(password)

    async def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate a user and return user object if valid."""
        user = self.db.query(User).filter(User.username == username).first()
        if not user or not self.verify_password(password, user.password_hash):
            return None
        return user

    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token."""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt

    async def get_current_user(self, token: str = Depends(oauth2_scheme)) -> User:
        """Get current user from JWT token."""
        credentials_exception = HTTPException(
            status_code=401,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username: str = payload.get("sub")
            if username is None:
                raise credentials_exception
        except JWTError:
            raise credentials_exception
            
        user = self.db.query(User).filter(User.username == username).first()
        if user is None:
            raise credentials_exception
        return user

class AccessControl:
    def __init__(self, db_session: Session):
        self.db = db_session

    def check_rbac_permission(self, user: User, required_permission: str) -> bool:
        """Check if user has required permission based on their role."""
        if not user.role:
            return False
        return required_permission in user.role.permissions

    def check_abac_permission(self, user: User, resource_attributes: Dict[str, Any]) -> bool:
        """
        Check if user has access based on attributes of both user and resource.
        """
        # Get user attributes
        user_attrs = user.attributes or {}
        
        # Basic ABAC rules
        # 1. Department-based access
        if resource_attributes.get('department'):
            if user_attrs.get('department') != resource_attributes['department']:
                return False
                
        # 2. Project-based access
        if resource_attributes.get('project'):
            user_projects = user_attrs.get('projects', [])
            if resource_attributes['project'] not in user_projects:
                return False
                
        # 3. Security clearance level
        user_clearance = user_attrs.get('security_clearance', 0)
        required_clearance = resource_attributes.get('required_clearance', 0)
        if user_clearance < required_clearance:
            return False
            
        return True

    async def filter_resources_by_permission(self, user: User, resources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter a list of resources based on user's permissions."""
        allowed_resources = []
        
        for resource in resources:
            # Check both RBAC and ABAC
            if (self.check_rbac_permission(user, f"view_{resource['type']}") and
                self.check_abac_permission(user, resource.get('attributes', {}))):
                allowed_resources.append(resource)
                
        return allowed_resources