import os
from cdp.auth.utils.jwt import generate_jwt, JwtOptions

# Validate environment variables
required_vars = ['KEY_NAME', 'KEY_SECRET', 'REQUEST_METHOD', 'REQUEST_HOST', 'REQUEST_PATH']
missing = [var for var in required_vars if not os.getenv(var)]
if missing:
    raise EnvironmentError(f"Missing environment variables: {', '.join(missing)}")

# Generate JWT
jwt_token = generate_jwt(JwtOptions(
    api_key_id=os.getenv('KEY_NAME'),
    api_key_secret=os.getenv('KEY_SECRET'),
    request_method=os.getenv('REQUEST_METHOD'),
    request_host=os.getenv('REQUEST_HOST'),
    request_path=os.getenv('REQUEST_PATH'),
    expires_in=120
))

# Only print the JWT
print(jwt_token)
