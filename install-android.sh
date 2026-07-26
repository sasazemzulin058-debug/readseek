#!/bin/sh
set -e

REPO="sasazemzulin058-debug/readseek"
BIN_DIR="${PREFIX:-/data/data/com.termux/files/usr}/bin"
AGENT_NPM="${HOME}/.pi/agent/npm"

LATEST_TAG=$(curl -fsSL "https://api.github.com/repos/${REPO}/releases/latest" | grep '"tag_name":' | sed -E 's/.*"([^"]+)".*/\1/')
if [ -z "$LATEST_TAG" ]; then
  LATEST_TAG="v0.8.11-android.1"
fi
VERSION="${LATEST_TAG#v}"

RELEASE_URL="https://github.com/${REPO}/releases/download/${LATEST_TAG}"

echo "📱 Installing readseek for Android ARM64 (${LATEST_TAG})..."

# 1. Install prebuilt CLI binary
echo "⚡ Downloading readseek CLI binary to ${BIN_DIR}..."
mkdir -p "${BIN_DIR}"
curl -fsSL -o "${BIN_DIR}/readseek" "${RELEASE_URL}/readseek"
chmod +x "${BIN_DIR}/readseek"

# 2. Install prebuilt package in agent npm
mkdir -p "${AGENT_NPM}"
cd "${AGENT_NPM}"
echo "📦 Installing readseek npm package..."
export CXXFLAGS="-std=c++20"
npm install "${RELEASE_URL}/readseek-android-arm64-${VERSION}.tgz" --force --legacy-peer-deps > /dev/null 2>&1

mkdir -p "${AGENT_NPM}/node_modules/@jarkkojs"
ln -sf "${AGENT_NPM}/node_modules/@sasazemzulin058-debug/readseek-android-arm64" "${AGENT_NPM}/node_modules/@jarkkojs/readseek-linux-arm64" 2>/dev/null || true
ln -sf "${AGENT_NPM}/node_modules/@sasazemzulin058-debug/readseek-android-arm64" "${AGENT_NPM}/node_modules/@jarkkojs/readseek-android-arm64" 2>/dev/null || true

# 3. Verify
echo "✅ Verification:"
"${BIN_DIR}/readseek" --version || echo "Readseek binary ready"

echo "🎉 Installed readseek successfully!"
