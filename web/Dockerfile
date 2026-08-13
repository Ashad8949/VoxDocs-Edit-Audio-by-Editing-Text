# Build the client, then serve it from nginx with /api proxied to the API server.
FROM node:22-slim AS build

WORKDIR /app
COPY package.json package-lock.json ./
COPY packages/edl/package.json ./packages/edl/
COPY services/api/package.json ./services/api/
COPY web/package.json ./web/
RUN npm ci --workspace @voxdocs/web --include-workspace-root

COPY packages/edl ./packages/edl
COPY web ./web
RUN npm run build --workspace @voxdocs/web

FROM nginx:1.27-alpine
COPY web/nginx.conf /etc/nginx/templates/default.conf.template
COPY --from=build /app/web/dist /usr/share/nginx/html
ENV VOXDOCS_API_HOST=voxdocs-api
ENV VOXDOCS_API_PORT=3000
EXPOSE 80
