"""
OpenAPI/Swagger Documentation.

Generates API documentation using Flask-RESTX or flasgger.
"""

from flask import Blueprint, jsonify
from typing import Dict, Any

# Try to use flasgger if available
try:
    from flasgger import Swagger, swag_from
    FLASGGER_AVAILABLE = True
except ImportError:
    FLASGGER_AVAILABLE = False
    try:
        from flask_restx import Api, Resource, fields
        RESTX_AVAILABLE = True
    except ImportError:
        RESTX_AVAILABLE = False

docs_bp = Blueprint('docs', __name__)


def get_openapi_spec() -> Dict[str, Any]:
    """
    Generate OpenAPI 3.0 specification for the API.
    
    Returns:
        OpenAPI specification dictionary
    """
    spec = {
        "openapi": "3.0.0",
        "info": {
            "title": "Trading Signals API",
            "version": "1.0.0",
            "description": "REST API for Trading Signals Platform",
            "contact": {
                "email": "support@tradingsignals.com"
            }
        },
        "servers": [
            {
                "url": "http://localhost:8050/api/v1",
                "description": "Development server"
            }
        ],
        "tags": [
            {"name": "Authentication", "description": "User authentication endpoints"},
            {"name": "Signals", "description": "Trading signals endpoints"},
            {"name": "Account", "description": "Account management endpoints"},
            {"name": "Subscriptions", "description": "Subscription management endpoints"},
            {"name": "Admin", "description": "Admin endpoints (requires admin role)"}
        ],
        "components": {
            "securitySchemes": {
                "bearerAuth": {
                    "type": "http",
                    "scheme": "bearer",
                    "bearerFormat": "JWT"
                },
                "apiKeyAuth": {
                    "type": "apiKey",
                    "in": "header",
                    "name": "X-API-Key"
                }
            },
            "schemas": {
                "Error": {
                    "type": "object",
                    "properties": {
                        "success": {"type": "boolean", "example": False},
                        "message": {"type": "string", "example": "Error message"},
                        "timestamp": {"type": "string", "format": "date-time"}
                    }
                },
                "Success": {
                    "type": "object",
                    "properties": {
                        "success": {"type": "boolean", "example": True},
                        "message": {"type": "string", "example": "Success message"},
                        "data": {"type": "object"},
                        "timestamp": {"type": "string", "format": "date-time"}
                    }
                },
                "User": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "format": "uuid"},
                        "email": {"type": "string", "format": "email"},
                        "full_name": {"type": "string"},
                        "email_verified": {"type": "boolean"},
                        "is_admin": {"type": "boolean"},
                        "is_active": {"type": "boolean"},
                        "tier": {"type": "string", "enum": ["free", "essential", "advanced"]},
                        "created_at": {"type": "string", "format": "date-time"},
                        "last_login": {"type": "string", "format": "date-time"}
                    }
                },
                "Signal": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "format": "uuid"},
                        "symbol": {"type": "string"},
                        "signal_type": {"type": "string", "enum": ["buy", "sell"]},
                        "confluence_score": {"type": "number", "format": "float"},
                        "technical_score": {"type": "number", "format": "float"},
                        "sentiment_score": {"type": "number", "format": "float"},
                        "ml_score": {"type": "number", "format": "float"},
                        "price_at_signal": {"type": "number", "format": "float"},
                        "risk_reward_ratio": {"type": "number", "format": "float"},
                        "var_95": {"type": "number", "format": "float"},
                        "suggested_position_size": {"type": "number", "format": "float"},
                        "created_at": {"type": "string", "format": "date-time"},
                        "technical_rationale": {"type": "string"},
                        "sentiment_rationale": {"type": "string"}
                    }
                },
                "LoginRequest": {
                    "type": "object",
                    "required": ["email", "password"],
                    "properties": {
                        "email": {"type": "string", "format": "email"},
                        "password": {"type": "string", "format": "password"}
                    }
                },
                "RegisterRequest": {
                    "type": "object",
                    "required": ["email", "password", "full_name"],
                    "properties": {
                        "email": {"type": "string", "format": "email"},
                        "password": {"type": "string", "format": "password"},
                        "full_name": {"type": "string"}
                    }
                }
            }
        },
        "paths": {
            "/auth/login": {
                "post": {
                    "tags": ["Authentication"],
                    "summary": "User login",
                    "description": "Authenticate user and return session",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/LoginRequest"}
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Login successful",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/Success"}
                                }
                            }
                        },
                        "401": {
                            "description": "Invalid credentials",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/Error"}
                                }
                            }
                        }
                    }
                }
            },
            "/auth/register": {
                "post": {
                    "tags": ["Authentication"],
                    "summary": "User registration",
                    "description": "Register a new user account",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/RegisterRequest"}
                            }
                        }
                    },
                    "responses": {
                        "201": {
                            "description": "Registration successful",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/Success"}
                                }
                            }
                        },
                        "400": {
                            "description": "Validation error",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/Error"}
                                }
                            }
                        }
                    }
                }
            },
            "/signals": {
                "get": {
                    "tags": ["Signals"],
                    "summary": "Get trading signals",
                    "description": "Retrieve trading signals with optional filtering",
                    "security": [{"bearerAuth": []}, {"apiKeyAuth": []}],
                    "parameters": [
                        {"name": "limit", "in": "query", "schema": {"type": "integer", "default": 50}},
                        {"name": "symbol", "in": "query", "schema": {"type": "string"}},
                        {"name": "min_confidence", "in": "query", "schema": {"type": "number"}},
                        {"name": "signal_type", "in": "query", "schema": {"type": "string", "enum": ["buy", "sell"]}},
                        {"name": "days", "in": "query", "schema": {"type": "integer"}}
                    ],
                    "responses": {
                        "200": {
                            "description": "List of signals",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "success": {"type": "boolean"},
                                            "data": {
                                                "type": "object",
                                                "properties": {
                                                    "signals": {
                                                        "type": "array",
                                                        "items": {"$ref": "#/components/schemas/Signal"}
                                                    },
                                                    "count": {"type": "integer"}
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/account": {
                "get": {
                    "tags": ["Account"],
                    "summary": "Get account information",
                    "description": "Get current user's account details",
                    "security": [{"bearerAuth": []}],
                    "responses": {
                        "200": {
                            "description": "Account information",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "success": {"type": "boolean"},
                                            "data": {
                                                "type": "object",
                                                "properties": {
                                                    "user": {"$ref": "#/components/schemas/User"}
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/account/api-keys": {
                "get": {
                    "tags": ["Account"],
                    "summary": "Get API keys",
                    "description": "List all API keys for current user",
                    "security": [{"bearerAuth": []}],
                    "responses": {
                        "200": {
                            "description": "List of API keys",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/Success"}
                                }
                            }
                        }
                    }
                },
                "post": {
                    "tags": ["Account"],
                    "summary": "Create API key",
                    "description": "Create a new API key",
                    "security": [{"bearerAuth": []}],
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"}
                                    }
                                }
                            }
                        }
                    },
                    "responses": {
                        "201": {
                            "description": "API key created",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/Success"}
                                }
                            }
                        }
                    }
                }
            },
            "/subscription": {
                "get": {
                    "tags": ["Subscriptions"],
                    "summary": "Get subscription",
                    "description": "Get current user's subscription details",
                    "security": [{"bearerAuth": []}],
                    "responses": {
                        "200": {
                            "description": "Subscription information",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/Success"}
                                }
                            }
                        }
                    }
                }
            },
            "/admin/audit-logs": {
                "get": {
                    "tags": ["Admin"],
                    "summary": "Get audit logs",
                    "description": "Get system audit logs (admin only)",
                    "security": [{"bearerAuth": []}],
                    "parameters": [
                        {"name": "limit", "in": "query", "schema": {"type": "integer", "default": 50}},
                        {"name": "user_id", "in": "query", "schema": {"type": "string", "format": "uuid"}},
                        {"name": "action", "in": "query", "schema": {"type": "string"}},
                        {"name": "days", "in": "query", "schema": {"type": "integer", "default": 7}}
                    ],
                    "responses": {
                        "200": {
                            "description": "Audit logs",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/Success"}
                                }
                            }
                        },
                        "403": {
                            "description": "Admin access required",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/Error"}
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    return spec


@docs_bp.route('/docs/openapi.json', methods=['GET'])
def get_openapi_json():
    """
    Serve OpenAPI specification as JSON.
    """
    spec = get_openapi_spec()
    return jsonify(spec)


@docs_bp.route('/docs', methods=['GET'])
def get_docs():
    """
    Serve API documentation page.
    
    Returns HTML page with Swagger UI if available, otherwise basic docs.
    """
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Trading Signals API Documentation</title>
        <link rel="stylesheet" type="text/css" href="https://unpkg.com/swagger-ui-dist@4.15.5/swagger-ui.css" />
        <style>
            body { margin: 0; }
        </style>
    </head>
    <body>
        <div id="swagger-ui"></div>
        <script src="https://unpkg.com/swagger-ui-dist@4.15.5/swagger-ui-bundle.js"></script>
        <script>
            window.onload = function() {
                const ui = SwaggerUIBundle({
                    url: '/api/v1/docs/openapi.json',
                    dom_id: '#swagger-ui',
                    presets: [
                        SwaggerUIBundle.presets.apis,
                        SwaggerUIBundle.presets.standalone
                    ]
                });
            };
        </script>
    </body>
    </html>
    """
    return html, 200, {'Content-Type': 'text/html'}
