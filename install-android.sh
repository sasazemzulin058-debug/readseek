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

echo "📱 Installing readseek & pi-readseek for Android ARM64 (${LATEST_TAG})..."

echo "⚡ Downloading readseek CLI binary to ${BIN_DIR}..."
mkdir -p "${BIN_DIR}"
curl -fsSL -o "${BIN_DIR}/readseek" "${RELEASE_URL}/readseek"
chmod +x "${BIN_DIR}/readseek"

echo "📦 Installing npm packages in ${AGENT_NPM}..."
mkdir -p "${AGENT_NPM}"
cd "${AGENT_NPM}"
export CXXFLAGS="-std=c++20"
npm install "${RELEASE_URL}/readseek-android-arm64-${VERSION}.tgz" "${RELEASE_URL}/pi-readseek-${VERSION}.tgz" --force --legacy-peer-deps > /dev/null 2>&1

echo "✅ Verification:"
"${BIN_DIR}/readseek" --version || echo "Readseek binary ready"

echo "🎉 Installed readseek and pi-readseek successfully!"
