# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the `@cubejs-backend/native` package - a native module that provides Rust-based bindings for Cube.js. It's a hybrid Node.js/Rust project that bridges JavaScript with high-performance Rust implementations for SQL processing and data operations.

## Architecture

### Core Components

- **JavaScript Layer** (`js/`): TypeScript interfaces and wrappers
  - `js/index.ts`: Main entry point with exports
  - `js/ResultWrapper.ts`: Result handling wrapper

- **Rust Core** (`src/`): Native implementation
  - `src/lib.rs`: Main library entry point
  - `src/gateway/`: HTTP server and routing logic
  - `src/python/`: Python runtime integration
  - `src/template/`: Jinja templating engine integration
  - `src/cross/`: Cross-language representation utilities

- **Python Integration** (`python/`): Python bindings and utilities
  - Supports Python 3.9-3.12 with dynamic library loading
  - Provides cube configuration loading and runtime execution

### Build System

The project uses a dual build system:
- **Rust**: Cargo for native compilation
- **Node.js**: TypeScript compilation and packaging

## Development Commands

### Building

```bash
# TypeScript compilation
yarn build
yarn tsc

# Native Rust builds
yarn native:build-debug          # Debug build (fallback, no Python)
yarn native:build-release        # Release build (fallback, no Python)
yarn native:build-debug-python   # Debug build with Python support
yarn native:build-release-python # Release build with Python support
```

### Testing

```bash
# Unit tests (Jest)
yarn test:unit
yarn unit

# Rust tests
yarn test:cargo

# Test servers for manual testing
yarn test:server        # Start test server with tracing
yarn test:server:stream # Start test server in stream mode
yarn test:python        # Test Python integration
```

### Linting

```bash
yarn lint           # ESLint for TypeScript files
yarn lint:fix       # Auto-fix linting issues
```

## Development Workflow

1. **Local Development Setup**:
   ```bash
   yarn native:build-debug  # or -python variant
   yarn link
   ```

2. **In Cube Project**:
   ```bash
   yarn link "@cubejs-backend/native"
   yarn dev
   ```

3. **After Changes**: Always rebuild the native module when modifying Rust code

## Key Technical Details

### Platform Support
- **With Python**: Linux x64/arm64 only
- **Fallback**: Linux, macOS, Windows (x64/arm64)

### Dependencies
- Requires `rustup` for development
- Uses Neon for Node.js/Rust bindings
- Optional PyO3 for Python integration
- Axum for HTTP server functionality

### Known Issues
- **macOS ARM**: `index.node` can become corrupted during development
  - Fix: `rm -rf index.node && yarn native:build && yarn test:unit`

## File Structure Notes

- `native/index.node`: Compiled native binary (generated)
- `target/`: Rust build artifacts
- `test/`: Jest test files and fixtures
- Tests require TypeScript compilation (`yarn build`) before running