# Frontend image cho FaceGuard (Next.js 16, chạy `next start`).
# Build context = webapp/frontend (frontend self-contained, không import ai_core).
#
# BACKEND_API_BASE_URL được đọc ở RUNTIME (các route force-dynamic), nên truyền qua
# env lúc chạy container — KHÔNG cần build-time. Phải là URL backend CÔNG KHAI vì
# nó vừa dùng proxy REST server-side, vừa nhúng vào ticket WebSocket cho browser.

FROM node:24-bookworm-slim AS base
RUN corepack enable && corepack prepare pnpm@latest --activate

# --- deps ---
FROM base AS deps
WORKDIR /app
COPY package.json pnpm-lock.yaml pnpm-workspace.yaml ./
RUN pnpm install --frozen-lockfile

# --- build ---
FROM base AS build
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN pnpm build

# --- run ---
FROM base AS run
WORKDIR /app
ENV NODE_ENV=production
COPY --from=build /app/node_modules ./node_modules
COPY --from=build /app/.next ./.next
COPY --from=build /app/public ./public
COPY --from=build /app/package.json ./package.json
COPY --from=build /app/next.config.mjs ./next.config.mjs
EXPOSE 3000
CMD ["pnpm", "start"]
