"""Security layer for RAG pipeline implementing RBAC and ABAC.

Features:
- Role-Based Access Control (RBAC)
- Attribute-Based Access Control (ABAC)
- Dynamic permission evaluation
- Integration with vector DB collections
"""

from __future__ import annotations

import datetime
import enum
import json
from typing import Dict, List, Optional, Set, Union

import jwt
from pydantic import BaseModel, Field

class Permission(str, enum.Enum):
    """Basic permission types."""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"

class Role(BaseModel):
    """Role definition with permissions."""
    name: str
    permissions: Set[Permission]
    attributes: Dict[str, List[str]] = Field(default_factory=dict)

class User(BaseModel):
    """User with roles and attributes."""
    id: str
    username: str
    roles: List[str]
    attributes: Dict[str, List[str]] = Field(default_factory=dict)

class AccessPolicy(BaseModel):
    """ABAC policy definition."""
    name: str
    effect: str = "allow"  # allow or deny
    conditions: Dict[str, List[str]]
    permissions: Set[Permission]

class SecurityManager:
    """Manages RBAC/ABAC security."""
    
    def __init__(self, jwt_secret: str):
        self.jwt_secret = jwt_secret
        self.roles: Dict[str, Role] = {}
        self.users: Dict[str, User] = {}
        self.policies: List[AccessPolicy] = []
    
    def add_role(self, role: Role):
        """Add a role definition."""
        self.roles[role.name] = role
    
    def add_user(self, user: User):
        """Add a user."""
        self.users[user.id] = user
    
    def add_policy(self, policy: AccessPolicy):
        """Add an ABAC policy."""
        self.policies.append(policy)
    
    def get_user_permissions(self, user_id: str) -> Set[Permission]:
        """Get all permissions for a user based on their roles."""
        if user_id not in self.users:
            return set()
        
        user = self.users[user_id]
        permissions = set()
        
        # Combine permissions from all roles
        for role_name in user.roles:
            if role_name in self.roles:
                permissions.update(self.roles[role_name].permissions)
        
        return permissions
    
    def _evaluate_condition(
        self,
        condition: Dict[str, List[str]],
        user_attributes: Dict[str, List[str]]
    ) -> bool:
        """Evaluate ABAC condition against user attributes."""
        for attr, values in condition.items():
            if attr not in user_attributes:
                return False
            if not any(v in user_attributes[attr] for v in values):
                return False
        return True
    
    def check_permission(
        self,
        user_id: str,
        required_permission: Permission,
        context: Optional[Dict] = None
    ) -> bool:
        """Check if user has permission in given context."""
        if user_id not in self.users:
            return False
        
        user = self.users[user_id]
        
        # Check RBAC permissions first
        permissions = self.get_user_permissions(user_id)
        if Permission.ADMIN in permissions:
            return True
        
        if required_permission in permissions:
            # User has basic permission, now check ABAC policies
            for policy in self.policies:
                if required_permission in policy.permissions:
                    if self._evaluate_condition(policy.conditions, user.attributes):
                        return policy.effect == "allow"
            
            # No ABAC policies matched, allow if user has basic permission
            return True
        
        return False
    
    def generate_token(self, user_id: str, expiry: int = 3600) -> str:
        """Generate JWT token for user."""
        if user_id not in self.users:
            raise ValueError(f"User {user_id} not found")
        
        user = self.users[user_id]
        payload = {
            "sub": user_id,
            "username": user.username,
            "roles": user.roles,
            "exp": datetime.datetime.utcnow() + datetime.timedelta(seconds=expiry)
        }
        
        return jwt.encode(payload, self.jwt_secret, algorithm="HS256")
    
    def validate_token(self, token: str) -> Optional[Dict]:
        """Validate JWT token and return payload."""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            return payload
        except jwt.InvalidTokenError:
            return None

class SecureVectorDB:
    """Security wrapper for vector database access."""
    
    def __init__(self, security_manager: SecurityManager):
        self.security = security_manager
        self._collection_policies: Dict[str, List[AccessPolicy]] = {}
    
    def add_collection_policy(
        self,
        collection_name: str,
        policy: AccessPolicy
    ):
        """Add a policy for a specific collection."""
        if collection_name not in self._collection_policies:
            self._collection_policies[collection_name] = []
        self._collection_policies[collection_name].append(policy)
    
    def check_collection_access(
        self,
        user_id: str,
        collection_name: str,
        permission: Permission
    ) -> bool:
        """Check if user can access collection."""
        # First check basic permission
        if not self.security.check_permission(user_id, permission):
            return False
        
        # Then check collection-specific policies
        if collection_name in self._collection_policies:
            user = self.security.users.get(user_id)
            if not user:
                return False
            
            for policy in self._collection_policies[collection_name]:
                if permission in policy.permissions:
                    if self.security._evaluate_condition(
                        policy.conditions,
                        user.attributes
                    ):
                        return policy.effect == "allow"
        
        # No collection-specific policies matched
        return True
    
    def filter_results(
        self,
        user_id: str,
        collection_name: str,
        results: List[Dict]
    ) -> List[Dict]:
        """Filter search results based on user permissions."""
        if not self.check_collection_access(user_id, collection_name, Permission.READ):
            return []
        
        user = self.security.users.get(user_id)
        if not user:
            return []
        
        filtered = []
        for result in results:
            # Check result-level attributes against user attributes
            metadata = result.get("metadata", {})
            required_attrs = metadata.get("required_attributes", {})
            
            can_access = True
            for attr, values in required_attrs.items():
                if attr not in user.attributes:
                    can_access = False
                    break
                if not any(v in user.attributes[attr] for v in values):
                    can_access = False
                    break
            
            if can_access:
                filtered.append(result)
        
        return filtered

# Example usage
if __name__ == "__main__":
    # Initialize security
    security = SecurityManager(jwt_secret="your-secret-key")
    
    # Define roles
    admin_role = Role(
        name="admin",
        permissions={Permission.ADMIN},
        attributes={"department": ["IT"]}
    )
    
    reader_role = Role(
        name="reader",
        permissions={Permission.READ},
        attributes={"department": ["Sales", "Marketing"]}
    )
    
    security.add_role(admin_role)
    security.add_role(reader_role)
    
    # Add users
    admin_user = User(
        id="1",
        username="admin",
        roles=["admin"],
        attributes={"department": ["IT"]}
    )
    
    sales_user = User(
        id="2",
        username="sales",
        roles=["reader"],
        attributes={"department": ["Sales"]}
    )
    
    security.add_user(admin_user)
    security.add_user(sales_user)
    
    # Add ABAC policy
    sales_policy = AccessPolicy(
        name="sales_data",
        effect="allow",
        conditions={"department": ["Sales"]},
        permissions={Permission.READ}
    )
    
    security.add_policy(sales_policy)
    
    # Create secure DB wrapper
    secure_db = SecureVectorDB(security)
    
    # Add collection policy
    collection_policy = AccessPolicy(
        name="sales_collection",
        effect="allow",
        conditions={"department": ["Sales"]},
        permissions={Permission.READ}
    )
    
    secure_db.add_collection_policy("sales_docs", collection_policy)
    
    # Example checks
    print("\nPermission checks:")
    print(f"Admin can read: {security.check_permission('1', Permission.READ)}")
    print(f"Sales can access sales collection: {secure_db.check_collection_access('2', 'sales_docs', Permission.READ)}")
    
    # Generate token
    token = security.generate_token("1")
    print(f"\nAdmin token: {token}")
    
    # Validate token
    payload = security.validate_token(token)
    print(f"\nToken payload: {json.dumps(payload, indent=2)}")