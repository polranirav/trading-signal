#!/bin/bash
# Frontend Setup Script for React + TypeScript + Vite

echo "Setting up React + TypeScript + Vite frontend..."

# Create React + TypeScript + Vite project
npm create vite@latest frontend -- --template react-ts

cd frontend

# Install dependencies
npm install

# Install routing
npm install react-router-dom

# Install API client
npm install axios

# Install Material-UI (MUI)
npm install @mui/material @emotion/react @emotion/styled
npm install @mui/icons-material

# Install state management (Zustand - lightweight)
npm install zustand

# Install date formatting
npm install date-fns

# Install chart library (recharts)
npm install recharts

# Install form validation
npm install react-hook-form @hookform/resolvers zod

echo "Frontend setup complete!"
echo "Run 'cd frontend && npm run dev' to start development server"
