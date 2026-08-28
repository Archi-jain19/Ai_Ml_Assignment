# ==============================================================================
# Frontend Dockerfile for FacetLens UI
# Multi-stage: Node.js 20 Alpine Builder -> Nginx Alpine Production Server
# ==============================================================================

# Stage 1: Build production bundle
FROM node:20-alpine AS builder

WORKDIR /app

# Copy dependency specifications
COPY package.json package-lock.json* ./

# Install dependencies cleanly
RUN npm ci

# Copy frontend source code
COPY . .

# Build production bundle to /app/dist
RUN npm run build

# Stage 2: Serve via high-performance Nginx with API reverse proxy
FROM nginx:alpine

# Copy built static assets
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy custom Nginx configuration with reverse proxy for /api/
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose web server port
EXPOSE 80

# Health check
HEALTHCHECK --interval=20s --timeout=3s --retries=3 \
    CMD wget -q --spider http://localhost:80/ || exit 1

# Start Nginx in foreground
CMD ["nginx", "-g", "daemon off;"]
