# Cube.js Docker Build Guide

This guide explains the different Docker build strategies available for Cube.js and when to use each one.

## Available Dockerfiles

### 1. `latest.Dockerfile` - Production (Published Packages)
**Use when:** You want to deploy the official published packages from npm.

**Characteristics:**
- Downloads pre-built packages from npm registry
- Smallest image size (~300-500 MB)
- Fastest build time (~5-10 minutes)
- Suitable for most production deployments
- Multi-platform support (linux/amd64, linux/arm64)

**Build command:**
```bash
cd packages/cubejs-docker
docker build -t cubejs/cube:latest -f latest.Dockerfile .
```

**Multi-platform build:**
```bash
docker buildx build --platform linux/amd64,linux/arm64 \
  -t cubejs/cube:latest -f latest.Dockerfile .
```

---

### 2. `dev.Dockerfile` - Development (Full Build)
**Use when:** You need a complete development environment for testing unreleased changes.

**Characteristics:**
- Builds everything from source
- Largest image size (~2-3 GB)
- Longest build time (~30-60 minutes)
- Includes all build tools (Rust, Java, etc.)
- NODE_ENV=development
- Not optimized for production use

**Build command:**
```bash
cd packages/cubejs-docker
docker build -t cubejs/cube:dev -f dev.Dockerfile ../../
```

**Note:** Must run from the cubejs-docker directory but build context is the monorepo root (`../../`).

---

### 3. `prod.Dockerfile` - Production (Local Build) ⭐ **NEW**
**Use when:** You have local changes that need to be built from source but want a production-optimized image.

**Characteristics:**
- Builds from monorepo source (includes your local changes)
- Optimized for production use
- Medium image size (~500-800 MB) - **60-70% smaller than dev.Dockerfile**
- Medium build time (~20-30 minutes)
- NODE_ENV=production
- Multi-stage build with minimal runtime dependencies
- Rust native modules built with `--release` flag

**Build command:**
```bash
cd packages/cubejs-docker
docker build -t cubejs/cube:prod -f prod.Dockerfile ../../
```

---

### 4. `latest-debian-jdk.Dockerfile` - Production with Java
**Use when:** You need JDBC drivers (e.g., for Databricks).

**Characteristics:**
- Similar to `latest.Dockerfile`
- Includes OpenJDK 17
- Runs as non-root user (cube:cube)
- Platform: linux/amd64 only

**Build command:**
```bash
cd packages/cubejs-docker
docker build -t cubejs/cube:latest-debian-jdk -f latest-debian-jdk.Dockerfile .
```

---

## Production Dockerfile (`prod.Dockerfile`) Details

### Multi-Stage Build Process

The prod.Dockerfile uses a 4-stage build process for optimal size and performance:

#### Stage 1: Base
- Sets up Node.js base image
- Installs Rust toolchain (nightly-2022-03-08)
- Installs build dependencies
- Copies all package.json files for dependency resolution

#### Stage 2: Build
- **NODE_ENV=production** for optimized builds
- Installs all dependencies (including devDependencies for building)
- Copies all Rust source code from `/rust` folder
- Copies all package source code
- **Builds Rust native modules with `--release` flag** (optimized binary)
- Compiles all TypeScript packages
- Removes node_modules before next stage

#### Stage 3: Production Dependencies
- Installs only production dependencies (`yarn install --prod`)
- Excludes all devDependencies
- Handles databricks-jdbc-driver workaround

#### Stage 4: Final Runtime Image
- Fresh Node.js base (no build tools)
- **Only runtime dependencies**: ca-certificates, libssl3, python3.11
- Copies built artifacts from build stage
- Copies production node_modules from dependencies stage
- **Copies optimized index.node** (native module)
- Sets up CLI binaries and symlinks
- **Final size optimized**: No Rust toolchain, no build tools, no dev dependencies

### Key Optimizations vs dev.Dockerfile

| Optimization | dev.Dockerfile | prod.Dockerfile | Impact |
|--------------|----------------|-----------------|---------|
| **Rust Build Mode** | Debug (default) | Release (`--release`) | 🚀 Faster runtime, smaller binary |
| **NODE_ENV** | development | production | 🚀 Optimized builds, tree-shaking |
| **Final Stage** | Includes build tools | Runtime only | 📦 60-70% smaller image |
| **Dependencies** | All (including dev) | Production only | 📦 Reduced size |
| **Build Tools in Final** | ✅ Rust, gcc, cmake, Java | ❌ None | 📦 Reduced size |
| **node_modules** | Full (with dev deps) | Prod only | 📦 Reduced size |

### Image Size Comparison

Typical sizes for x86_64/amd64:

| Dockerfile | Compressed | Uncompressed |
|-----------|------------|--------------|
| latest.Dockerfile | ~300 MB | ~800 MB |
| dev.Dockerfile | ~900 MB | ~2.5 GB |
| **prod.Dockerfile** | **~400 MB** | **~1 GB** |
| latest-debian-jdk.Dockerfile | ~500 MB | ~1.2 GB |

---

## Building with Local Changes

### Scenario: You modified Rust code and TypeScript code

The `prod.Dockerfile` is perfect for this scenario:

```bash
# 1. Make your changes in the monorepo
# - Edit files in /rust folder (Rust native modules)
# - Edit files in /packages folder (TypeScript packages)

# 2. Build the production image with your changes
cd packages/cubejs-docker
docker build -t my-custom-cube:v1 -f prod.Dockerfile ../../

# 3. Run the container
docker run -p 4000:4000 \
  -v $(pwd)/cube-config:/cube/conf \
  my-custom-cube:v1
```

### Build Context Important Note

The build context for `prod.Dockerfile` is the **monorepo root** (`../../`), not the current directory. This is necessary because:
- Rust source code is in `/rust` (two levels up)
- Package source code is in `/packages` (two levels up)
- The Dockerfile needs access to the entire monorepo structure

---

## Native Module (index.node) Compilation

### What is index.node?

`index.node` is a compiled Rust native Node.js addon that provides:
- SQL interface (CubeSQL)
- Python integration support
- Query result transformation
- Jinja templating engine
- High-performance operations

### How it's built in prod.Dockerfile

```dockerfile
# In build stage
RUN cd packages/cubejs-backend-native && \
    yarn run native:build-release && \
    ls -lh index.node
```

This executes the build script from `packages/cubejs-backend-native/package.json`:

```json
{
  "native:build-release": "cargo-cp-artifact -a cdylib cubejs-native index.node -- cargo build --message-format=json-render-diagnostics --release"
}
```

The `--release` flag tells Cargo (Rust compiler) to build with optimizations:
- Aggressive optimizations enabled
- Debug symbols stripped
- Smaller binary size (typically 50-70% smaller than debug builds)
- Better runtime performance

### Where it's placed in final image

```dockerfile
# In final stage
COPY --from=build /cubejs/packages/cubejs-backend-native/index.node \
                  /cubejs/packages/cubejs-backend-native/index.node
```

The module is loaded at runtime by the loader in `packages/cubejs-backend-native/js/index.ts`.

---

## Multi-Platform Builds

### Building for ARM64 and AMD64

You can build the prod.Dockerfile for multiple platforms using buildx:

```bash
# Create a builder instance (one-time setup)
docker buildx create --name multiplatform --use

# Build for both platforms
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t my-custom-cube:v1 \
  -f prod.Dockerfile \
  --push \
  ../../
```

**Note:** ARM64 builds take longer due to native compilation. Consider:
- Using GitHub Actions with native ARM64 runners
- Cross-compilation (requires additional setup)
- Building on platform-specific machines

---

## Build Time Optimization Tips

### 1. Use BuildKit
```bash
export DOCKER_BUILDKIT=1
docker build -t cubejs/cube:prod -f prod.Dockerfile ../../
```

### 2. Layer Caching
The Dockerfile is structured to maximize layer caching:
- Dependencies are installed before copying source code
- package.json files are copied separately from source
- Rust compilation happens in a dedicated layer

### 3. Parallel Builds
If building for multiple platforms, use GitHub Actions or buildx with parallel workers.

### 4. Cache Mounts (BuildKit)
Add to Dockerfile stages:
```dockerfile
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/cubejs/target \
    cd packages/cubejs-backend-native && yarn run native:build-release
```

---

## Troubleshooting

### Build fails with "Rust toolchain not found"
**Solution:** The Rust installation in the base stage may have failed. Check your internet connection and try again.

### Build fails with "Cannot find module './index.node'"
**Solution:** The native module build failed. Check the build logs for Rust compilation errors. Ensure you're copying the entire `/rust` folder.

### Image size is larger than expected
**Possible causes:**
1. node_modules contains devDependencies - check Stage 3 logs
2. Source code was copied to final stage - verify COPY commands in final stage
3. Build tools remained in final stage - check apt-get commands in final stage

### Runtime error: "Your system is not supported by @cubejs-backend/native"
**Solution:** The index.node file is platform-specific. Ensure you're building for the correct platform (linux/amd64 or linux/arm64).

---

## Comparison Matrix

| Feature | latest.Dockerfile | dev.Dockerfile | **prod.Dockerfile** | latest-debian-jdk.Dockerfile |
|---------|------------------|----------------|---------------------|------------------------------|
| **Source** | npm packages | Monorepo | Monorepo | npm packages |
| **Build Time** | Fast (5-10 min) | Slow (30-60 min) | **Medium (20-30 min)** | Fast (5-10 min) |
| **Image Size** | Small (~300 MB) | Large (~900 MB) | **Medium (~400 MB)** | Medium (~500 MB) |
| **NODE_ENV** | production | development | **production** | production |
| **Rust Build** | Pre-built | Debug | **Release (optimized)** | Pre-built |
| **Build Tools in Final** | ❌ | ✅ | **❌** | ❌ |
| **Local Changes** | ❌ | ✅ | **✅** | ❌ |
| **Multi-platform** | ✅ | ⚠️ | **✅** | ❌ (amd64 only) |
| **Java/JDK** | ❌ | ✅ | **❌** | ✅ |
| **Best For** | Official releases | Development/Testing | **Custom production builds** | JDBC drivers |

---

## Recommended Workflow

### For Development
```bash
# Use dev.Dockerfile for full dev environment
docker build -t cubejs/cube:dev -f dev.Dockerfile ../../
```

### For Production with Local Changes
```bash
# Use prod.Dockerfile for optimized builds
docker build -t cubejs/cube:prod -f prod.Dockerfile ../../
```

### For Production with Published Packages
```bash
# Use latest.Dockerfile for smallest image
docker build -t cubejs/cube:latest -f latest.Dockerfile .
```

### For Production with JDBC Drivers
```bash
# Use latest-debian-jdk.Dockerfile
docker build -t cubejs/cube:latest-debian-jdk -f latest-debian-jdk.Dockerfile .
```

---

## Additional Resources

- [Cube.js Documentation](https://cube.dev/docs)
- [Docker Multi-Stage Builds](https://docs.docker.com/build/building/multi-stage/)
- [Rust Cargo Book](https://doc.rust-lang.org/cargo/)
- [Node.js Native Addons](https://nodejs.org/api/addons.html)

---

**Created:** 2025-10-28
**For:** Production-optimized Docker builds with local source changes
**Maintainer:** Cube.js Team
