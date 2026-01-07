# Moved to src/auth_utils.py

import os
import json
from fastapi import Depends, HTTPException, Request, status
from jose import jwt, JWTError
import requests
from dotenv import load_dotenv

load_dotenv()

AUTH0_DOMAIN = os.environ.get("AUTH0_DOMAIN")
AUTH0_AUDIENCE = os.environ.get("AUTH0_AUDIENCE")
ALGORITHMS = os.environ.get("ALGORITHMS", "RS256")

if not AUTH0_DOMAIN or not AUTH0_AUDIENCE:
    raise Exception("AUTH0_DOMAIN and AUTH0_AUDIENCE must be set")

# URL for JWKS from Auth0
JWKS_URL = f"https://{AUTH0_DOMAIN}/.well-known/jwks.json"

# Cache the JWKS instead of refetching every request
jwks = requests.get(JWKS_URL).json()

def get_token_auth_header(request: Request):
    """Extract token from Authorization header"""
    auth = request.headers.get("Authorization")
    if not auth:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Authorization header missing")
    parts = auth.split()
    if parts[0].lower() != "bearer":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Authorization header must start with Bearer")
    if len(parts) == 1:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Token not found")
    return parts[1]

def verify_jwt(token: str):
    """Verify the JWT using Auth0 JWKS + python-jose"""
    unverified_header = jwt.get_unverified_header(token)

    rsa_key = {}
    for key in jwks["keys"]:
        if key["kid"] == unverified_header.get("kid"):
            rsa_key = {
                "kty": key["kty"],
                "kid": key["kid"],
                "use": key["use"],
                "n": key["n"],
                "e": key["e"]
            }
    if not rsa_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Unable to find appropriate key")

    try:
        payload = jwt.decode(
            token,
            rsa_key,
            algorithms=[ALGORITHMS],
            audience=AUTH0_AUDIENCE,
            issuer=f"https://{AUTH0_DOMAIN}/"
        )
        return payload

    except JWTError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail=f"Invalid token: {str(e)}")

def get_current_user(request: Request):
    token = get_token_auth_header(request)
    return verify_jwt(token)
# auth_utils.py

import os
import json
from fastapi import Depends, HTTPException, Request, status
from jose import jwt, JWTError
import requests
from dotenv import load_dotenv

load_dotenv()

AUTH0_DOMAIN = os.environ.get("AUTH0_DOMAIN")
AUTH0_AUDIENCE = os.environ.get("AUTH0_AUDIENCE")
ALGORITHMS = os.environ.get("ALGORITHMS", "RS256")

if not AUTH0_DOMAIN or not AUTH0_AUDIENCE:
    raise Exception("AUTH0_DOMAIN and AUTH0_AUDIENCE must be set")

# URL for JWKS from Auth0
JWKS_URL = f"https://{AUTH0_DOMAIN}/.well-known/jwks.json"

# Cache the JWKS instead of refetching every request
jwks = requests.get(JWKS_URL).json()

def get_token_auth_header(request: Request):
    """Extract token from Authorization header"""
    auth = request.headers.get("Authorization")
    if not auth:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Authorization header missing")
    parts = auth.split()
    if parts[0].lower() != "bearer":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Authorization header must start with Bearer")
    if len(parts) == 1:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Token not found")
    return parts[1]

def verify_jwt(token: str):
    """Verify the JWT using Auth0 JWKS + python-jose"""
    unverified_header = jwt.get_unverified_header(token)

    rsa_key = {}
    for key in jwks["keys"]:
        if key["kid"] == unverified_header.get("kid"):
            rsa_key = {
                "kty": key["kty"],
                "kid": key["kid"],
                "use": key["use"],
                "n": key["n"],
                "e": key["e"]
            }
    if not rsa_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Unable to find appropriate key")

    try:
        payload = jwt.decode(
            token,
            rsa_key,
            algorithms=[ALGORITHMS],
            audience=AUTH0_AUDIENCE,
            issuer=f"https://{AUTH0_DOMAIN}/"
        )
        return payload

    except JWTError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail=f"Invalid token: {str(e)}")

def get_current_user(request: Request):
    token = get_token_auth_header(request)
    return verify_jwt(token)
