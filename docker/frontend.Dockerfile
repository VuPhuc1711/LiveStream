FROM node:22.20.0-bookworm-slim

WORKDIR /app/frontend

COPY docker/frontend-entrypoint.sh /usr/local/bin/frontend-entrypoint
RUN sed -i 's/\r$//' /usr/local/bin/frontend-entrypoint \
    && chmod 755 /usr/local/bin/frontend-entrypoint \
    && chown node:node /app/frontend

USER node
COPY --chown=node:node frontend/package.json frontend/package-lock.json ./
# Also seeds the marker copied into a fresh node_modules named volume.
RUN frontend-entrypoint true

COPY --chown=node:node frontend/ ./

EXPOSE 5173
ENTRYPOINT ["frontend-entrypoint"]
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "5173", "--strictPort"]
