# The API server: ffmpeg, disk and sockets. No model, so it stays small and
# scales on request volume rather than on CPU.
FROM node:22-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Workspace manifests first, so npm ci caches independently of source changes.
COPY package.json package-lock.json ./
COPY packages/edl/package.json ./packages/edl/
COPY services/api/package.json ./services/api/
COPY web/package.json ./web/

RUN npm ci --omit=dev --workspace @voxdocs/api --include-workspace-root

COPY packages/edl ./packages/edl
COPY services/api ./services/api

ENV NODE_ENV=production
ENV PORT=3000
ENV VOXDOCS_DATA_DIR=/data
EXPOSE 3000
VOLUME ["/data"]

CMD ["node", "services/api/src/index.js"]
