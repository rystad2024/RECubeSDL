# Production-optimized Dockerfile for building Cube.js from source
# This builds all packages from the monorepo including Rust native modules
# while optimizing for smaller image size and runtime performance.
#
# Key optimizations over prod.Dockerfile:
#   - Removes ~1GB Rust target/ build artifacts from final image
#   - Removes Rust source code not needed at runtime
#   - Strips debug symbols from native binaries (~55M -> ~15M)
#   - Orders COPY layers to avoid stale build artifact bleed-through
#
# Usage:
#   docker build -t cubejs/cube:prod -f prod-optimal.Dockerfile ../../

FROM node:22.16.0-bookworm-slim AS base

ARG IMAGE_VERSION=prod

ENV CUBEJS_DOCKER_IMAGE_VERSION=$IMAGE_VERSION
ENV CUBEJS_DOCKER_IMAGE_TAG=prod
ENV CI=0

# Install build dependencies (Rust toolchain + build tools)
RUN DEBIAN_FRONTEND=noninteractive apt-get update \
    && apt-get install -y --no-install-recommends \
       libssl3 curl ca-certificates \
       cmake python3 python3.11 libpython3.11-dev \
       gcc g++ make \
    && rm -rf /var/lib/apt/lists/*

# Install Rust toolchain for building native modules
# Use 1.84.1 to match rust-toolchain.toml in cubejs-backend-native.
# The official CI uses 1.90.0 but rust-toolchain.toml pins 1.84.1.
ENV RUSTUP_HOME=/usr/local/rustup
ENV CARGO_HOME=/usr/local/cargo
ENV PATH=/usr/local/cargo/bin:$PATH

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
    sh -s -- --profile minimal --default-toolchain 1.84.1 -y

# Skip cubestore post-install (we're building from source)
ENV CUBESTORE_SKIP_POST_INSTALL=true

WORKDIR /cubejs

# Setup Yarn
RUN yarn policies set-version v1.22.22
RUN yarn config set network-timeout 120000 -g

# Copy root package files
COPY package.json .
COPY lerna.json .
COPY yarn.lock .
COPY tsconfig.base.json .
COPY rollup.config.js .
COPY packages/cubejs-linter packages/cubejs-linter

# Copy all package.json files for dependency resolution
# Backend packages
COPY rust/cubesql/package.json rust/cubesql/package.json
COPY rust/cubestore/package.json rust/cubestore/package.json
COPY rust/cubestore/bin rust/cubestore/bin
COPY packages/cubejs-backend-shared/package.json packages/cubejs-backend-shared/package.json
COPY packages/cubejs-base-driver/package.json packages/cubejs-base-driver/package.json
COPY packages/cubejs-backend-native/package.json packages/cubejs-backend-native/package.json
COPY packages/cubejs-testing-shared/package.json packages/cubejs-testing-shared/package.json
COPY packages/cubejs-backend-cloud/package.json packages/cubejs-backend-cloud/package.json
COPY packages/cubejs-api-gateway/package.json packages/cubejs-api-gateway/package.json
COPY packages/cubejs-athena-driver/package.json packages/cubejs-athena-driver/package.json
COPY packages/cubejs-bigquery-driver/package.json packages/cubejs-bigquery-driver/package.json
COPY packages/cubejs-cli/package.json packages/cubejs-cli/package.json
COPY packages/cubejs-clickhouse-driver/package.json packages/cubejs-clickhouse-driver/package.json
COPY packages/cubejs-crate-driver/package.json packages/cubejs-crate-driver/package.json
COPY packages/cubejs-dremio-driver/package.json packages/cubejs-dremio-driver/package.json
COPY packages/cubejs-druid-driver/package.json packages/cubejs-druid-driver/package.json
COPY packages/cubejs-duckdb-driver/package.json packages/cubejs-duckdb-driver/package.json
COPY packages/cubejs-elasticsearch-driver/package.json packages/cubejs-elasticsearch-driver/package.json
COPY packages/cubejs-firebolt-driver/package.json packages/cubejs-firebolt-driver/package.json
COPY packages/cubejs-hive-driver/package.json packages/cubejs-hive-driver/package.json
COPY packages/cubejs-mongobi-driver/package.json packages/cubejs-mongobi-driver/package.json
COPY packages/cubejs-mssql-driver/package.json packages/cubejs-mssql-driver/package.json
COPY packages/cubejs-mysql-driver/package.json packages/cubejs-mysql-driver/package.json
COPY packages/cubejs-cubestore-driver/package.json packages/cubejs-cubestore-driver/package.json
COPY packages/cubejs-oracle-driver/package.json packages/cubejs-oracle-driver/package.json
COPY packages/cubejs-redshift-driver/package.json packages/cubejs-redshift-driver/package.json
COPY packages/cubejs-postgres-driver/package.json packages/cubejs-postgres-driver/package.json
COPY packages/cubejs-questdb-driver/package.json packages/cubejs-questdb-driver/package.json
COPY packages/cubejs-materialize-driver/package.json packages/cubejs-materialize-driver/package.json
COPY packages/cubejs-prestodb-driver/package.json packages/cubejs-prestodb-driver/package.json
COPY packages/cubejs-trino-driver/package.json packages/cubejs-trino-driver/package.json
COPY packages/cubejs-pinot-driver/package.json packages/cubejs-pinot-driver/package.json
COPY packages/cubejs-query-orchestrator/package.json packages/cubejs-query-orchestrator/package.json
COPY packages/cubejs-schema-compiler/package.json packages/cubejs-schema-compiler/package.json
COPY packages/cubejs-server/package.json packages/cubejs-server/package.json
COPY packages/cubejs-server-core/package.json packages/cubejs-server-core/package.json
COPY packages/cubejs-snowflake-driver/package.json packages/cubejs-snowflake-driver/package.json
COPY packages/cubejs-sqlite-driver/package.json packages/cubejs-sqlite-driver/package.json
COPY packages/cubejs-ksql-driver/package.json packages/cubejs-ksql-driver/package.json
COPY packages/cubejs-dbt-schema-extension/package.json packages/cubejs-dbt-schema-extension/package.json
COPY packages/cubejs-jdbc-driver/package.json packages/cubejs-jdbc-driver/package.json
COPY packages/cubejs-databricks-jdbc-driver/package.json packages/cubejs-databricks-jdbc-driver/package.json
COPY packages/cubejs-vertica-driver/package.json packages/cubejs-vertica-driver/package.json

# Workaround for databricks-jdbc-driver post-install script
# Create dummy post-install to prevent errors during yarn install
RUN mkdir -p packages/cubejs-databricks-jdbc-driver/bin && \
    echo '#!/usr/bin/env node' > packages/cubejs-databricks-jdbc-driver/bin/post-install && \
    chmod +x packages/cubejs-databricks-jdbc-driver/bin/post-install

# Frontend packages
COPY packages/cubejs-templates/package.json packages/cubejs-templates/package.json
COPY packages/cubejs-client-core/package.json packages/cubejs-client-core/package.json
COPY packages/cubejs-client-react/package.json packages/cubejs-client-react/package.json
COPY packages/cubejs-client-vue/package.json packages/cubejs-client-vue/package.json
COPY packages/cubejs-client-vue3/package.json packages/cubejs-client-vue3/package.json
COPY packages/cubejs-client-ngx/package.json packages/cubejs-client-ngx/package.json
COPY packages/cubejs-client-ws-transport/package.json packages/cubejs-client-ws-transport/package.json
COPY packages/cubejs-playground/package.json packages/cubejs-playground/package.json

# ==============================================================================
# Stage: Build all packages with production optimizations
# ==============================================================================
FROM base AS build

# Use production environment for optimized builds
ENV NODE_ENV=production

# Install all dependencies (including devDependencies needed for build)
# Use --production=false to force installation of devDependencies despite NODE_ENV=production
# This gives us build tools (cargo-cp-artifact, tsc) while keeping production optimizations
RUN yarn install --production=false

# Copy Rust source code for native module compilation
COPY rust/ rust/

# Copy all package source code
# Backend packages
COPY packages/cubejs-backend-shared/ packages/cubejs-backend-shared/
COPY packages/cubejs-base-driver/ packages/cubejs-base-driver/
COPY packages/cubejs-backend-native/ packages/cubejs-backend-native/
COPY packages/cubejs-testing-shared/ packages/cubejs-testing-shared/
COPY packages/cubejs-backend-cloud/ packages/cubejs-backend-cloud/
COPY packages/cubejs-api-gateway/ packages/cubejs-api-gateway/
COPY packages/cubejs-athena-driver/ packages/cubejs-athena-driver/
COPY packages/cubejs-bigquery-driver/ packages/cubejs-bigquery-driver/
COPY packages/cubejs-cli/ packages/cubejs-cli/
COPY packages/cubejs-clickhouse-driver/ packages/cubejs-clickhouse-driver/
COPY packages/cubejs-crate-driver/ packages/cubejs-crate-driver/
COPY packages/cubejs-dremio-driver/ packages/cubejs-dremio-driver/
COPY packages/cubejs-druid-driver/ packages/cubejs-druid-driver/
COPY packages/cubejs-duckdb-driver/ packages/cubejs-duckdb-driver/
COPY packages/cubejs-elasticsearch-driver/ packages/cubejs-elasticsearch-driver/
COPY packages/cubejs-firebolt-driver/ packages/cubejs-firebolt-driver/
COPY packages/cubejs-hive-driver/ packages/cubejs-hive-driver/
COPY packages/cubejs-mongobi-driver/ packages/cubejs-mongobi-driver/
COPY packages/cubejs-mssql-driver/ packages/cubejs-mssql-driver/
COPY packages/cubejs-mysql-driver/ packages/cubejs-mysql-driver/
COPY packages/cubejs-cubestore-driver/ packages/cubejs-cubestore-driver/
COPY packages/cubejs-oracle-driver/ packages/cubejs-oracle-driver/
COPY packages/cubejs-redshift-driver/ packages/cubejs-redshift-driver/
COPY packages/cubejs-postgres-driver/ packages/cubejs-postgres-driver/
COPY packages/cubejs-questdb-driver/ packages/cubejs-questdb-driver/
COPY packages/cubejs-materialize-driver/ packages/cubejs-materialize-driver/
COPY packages/cubejs-prestodb-driver/ packages/cubejs-prestodb-driver/
COPY packages/cubejs-trino-driver/ packages/cubejs-trino-driver/
COPY packages/cubejs-pinot-driver/ packages/cubejs-pinot-driver/
COPY packages/cubejs-query-orchestrator/ packages/cubejs-query-orchestrator/
COPY packages/cubejs-schema-compiler/ packages/cubejs-schema-compiler/
COPY packages/cubejs-server/ packages/cubejs-server/
COPY packages/cubejs-server-core/ packages/cubejs-server-core/
COPY packages/cubejs-snowflake-driver/ packages/cubejs-snowflake-driver/
COPY packages/cubejs-sqlite-driver/ packages/cubejs-sqlite-driver/
COPY packages/cubejs-ksql-driver/ packages/cubejs-ksql-driver/
COPY packages/cubejs-dbt-schema-extension/ packages/cubejs-dbt-schema-extension/
COPY packages/cubejs-jdbc-driver/ packages/cubejs-jdbc-driver/
COPY packages/cubejs-databricks-jdbc-driver/ packages/cubejs-databricks-jdbc-driver/
COPY packages/cubejs-vertica-driver/ packages/cubejs-vertica-driver/

# Frontend packages
COPY packages/cubejs-templates/ packages/cubejs-templates/
COPY packages/cubejs-client-core/ packages/cubejs-client-core/
COPY packages/cubejs-client-react/ packages/cubejs-client-react/
COPY packages/cubejs-client-vue/ packages/cubejs-client-vue/
COPY packages/cubejs-client-vue3/ packages/cubejs-client-vue3/
COPY packages/cubejs-client-ngx/ packages/cubejs-client-ngx/
COPY packages/cubejs-client-ws-transport/ packages/cubejs-client-ws-transport/
COPY packages/cubejs-playground/ packages/cubejs-playground/

# Build native Rust modules with release optimizations
# This compiles the Rust code from /rust and creates the optimized index.node
# Note: cargo-cp-artifact comes from devDependencies in package.json
RUN cd packages/cubejs-backend-native && \
    yarn native:build-release && \
    ls -lh index.node

# Build all packages:
# 1. "yarn build" builds cubejs-client-core via rollup — required for playground and other
#    @cubejs-client/* packages to compile (they import types from client-core).
# 2. "yarn lerna run build" compiles TypeScript for all remaining packages.
RUN yarn build
RUN yarn lerna run build

# ==============================================================================
# Cleanup build artifacts to minimize what gets copied to final stage
# ==============================================================================
# Remove Rust target/ directory (~986MB of intermediate build artifacts)
# The compiled index.node is already at packages/cubejs-backend-native/index.node
RUN rm -rf packages/cubejs-backend-native/target

# Strip debug symbols from the native binary (~55M -> ~15M)
RUN strip packages/cubejs-backend-native/index.node && \
    ls -lh packages/cubejs-backend-native/index.node

# Remove node_modules to prepare for production reinstall
RUN find . -name 'node_modules' -type d -prune -exec rm -rf '{}' +

# Remove Rust source code no longer needed (keep cubestore runtime: bin/, dist/, package.json)
RUN find rust/ -mindepth 1 -maxdepth 1 ! -name 'cubestore' -exec rm -rf {} + && \
    find rust/cubestore/ -mindepth 1 -maxdepth 1 ! -name 'bin' ! -name 'dist' ! -name 'package.json' -exec rm -rf {} +

# Remove test files, development configs, and documentation from packages
RUN find packages/ -type d -name 'test' -prune -exec rm -rf {} + 2>/dev/null; \
    find packages/ -type d -name '__tests__' -prune -exec rm -rf {} + 2>/dev/null; \
    find packages/ -name '*.md' ! -name 'README.md' -delete 2>/dev/null; \
    find packages/ -name 'tsconfig.json' -delete 2>/dev/null; \
    find packages/ -name '.eslintrc.*' -delete 2>/dev/null; \
    find packages/ -name 'jest.config.*' -delete 2>/dev/null; \
    true

# ==============================================================================
# Stage: Install production dependencies only
# ==============================================================================
FROM base AS prod_dependencies

# Install production dependencies only (ignore scripts to avoid post-install issues)
RUN yarn install --prod --ignore-scripts

# Copy the actual databricks scripts for runtime (needed even though we used --ignore-scripts)
COPY packages/cubejs-databricks-jdbc-driver/bin packages/cubejs-databricks-jdbc-driver/bin

# ==============================================================================
# Stage: Final runtime image (minimal and optimized)
# ==============================================================================
FROM node:22.16.0-bookworm-slim AS final

ARG IMAGE_VERSION=prod

ENV CUBEJS_DOCKER_IMAGE_VERSION=$IMAGE_VERSION
ENV CUBEJS_DOCKER_IMAGE_TAG=prod
ENV NODE_ENV=production
ENV PYTHONUNBUFFERED=1

# Install only runtime dependencies (no build tools)
RUN DEBIAN_FRONTEND=noninteractive apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates \
       libssl3 \
       python3.11 \
       libpython3.11-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /cubejs

# Setup Yarn
RUN yarn policies set-version v1.22.22

# Copy production node_modules FIRST (clean, no build artifacts)
COPY --from=prod_dependencies /cubejs/node_modules ./node_modules
COPY --from=prod_dependencies /cubejs/packages ./packages

# Copy built package artifacts (dist/, src/, etc.) from build stage ON TOP
# This overlays the compiled output onto the prod dependency tree.
# Because we cleaned target/ and node_modules in the build stage,
# this only adds the built JS/TS output — no bloat.
COPY --from=build /cubejs/packages ./packages

# Copy cubestore runtime files (bin scripts, built JS, package.json — not Rust source)
COPY --from=build /cubejs/rust/cubestore/bin ./rust/cubestore/bin
COPY --from=build /cubejs/rust/cubestore/dist ./rust/cubestore/dist
COPY --from=build /cubejs/rust/cubestore/package.json ./rust/cubestore/package.json

# Copy root configuration files
COPY --from=build /cubejs/package.json .
COPY --from=build /cubejs/lerna.json .
COPY --from=build /cubejs/yarn.lock .
COPY --from=build /cubejs/tsconfig.base.json .

# Remove DuckDB sources (not needed at runtime, matches official image)
RUN rm -rf node_modules/duckdb/src

# Setup CLI binaries
COPY packages/cubejs-docker/bin/cubejs-dev /usr/local/bin/cubejs
RUN ln -s /cubejs/packages/cubejs-docker /cube
RUN ln -s /cubejs/rust/cubestore/bin/cubestore-dev /usr/local/bin/cubestore-dev

# Setup Node path for module resolution
# /cube is a symlink to /cubejs/packages/cubejs-docker, so /cube/node_modules
# would resolve to a non-existent per-package dir. Point to the actual root
# node_modules at /cubejs/node_modules instead.
ENV NODE_PATH=/cube/conf/node_modules:/cubejs/node_modules

WORKDIR /cube/conf

EXPOSE 4000

CMD ["cubejs", "server"]
