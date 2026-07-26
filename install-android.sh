#!/bin/sh
set -e

REPO="sasazemzulin058-debug/readseek"
PREFIX_DIR="${PREFIX:-/data/data/com.termux/files/usr}"
BIN_DIR="${PREFIX_DIR}/bin"
AGENT_NPM="${HOME}/.pi/agent/npm"
TMPDIR_BASE="${TMPDIR:-${PREFIX_DIR}/tmp}"

mkdir -p "$TMPDIR_BASE"
TMP_WORK_DIR=$(mktemp -d -p "$TMPDIR_BASE" 2>/dev/null || mktemp -d)
trap 'rm -rf "$TMP_WORK_DIR"' EXIT INT TERM

LATEST_TAG=$(curl -fsSL -L --retry 3 "https://api.github.com/repos/${REPO}/releases/latest" | grep '"tag_name":' | sed -E 's/.*"([^"]+)".*/\1/')
if [ -z "$LATEST_TAG" ]; then
  LATEST_TAG="v0.8.11-android.1"
fi
VERSION="${LATEST_TAG#v}"
RELEASE_URL="https://github.com/${REPO}/releases/download/${LATEST_TAG}"

echo "📱 Installing readseek & pi-readseek for Android ARM64 (${LATEST_TAG})..."

echo "⚡ Downloading readseek CLI binary to ${BIN_DIR}..."
mkdir -p "${BIN_DIR}"
curl -fsSL -L --retry 3 -o "${TMP_WORK_DIR}/readseek" "${RELEASE_URL}/readseek"
mv "${TMP_WORK_DIR}/readseek" "${BIN_DIR}/readseek"
chmod +x "${BIN_DIR}/readseek"

echo "📦 Downloading and installing npm packages in ${AGENT_NPM}..."
mkdir -p "${AGENT_NPM}"
curl -fsSL -L --retry 3 -o "${TMP_WORK_DIR}/readseek-android-arm64.tgz" "${RELEASE_URL}/readseek-android-arm64-${VERSION}.tgz"
curl -fsSL -L --retry 3 -o "${TMP_WORK_DIR}/pi-readseek.tgz" "${RELEASE_URL}/pi-readseek-${VERSION}.tgz"

cd "${AGENT_NPM}"
export CXXFLAGS="-std=c++20"
npm install "${TMP_WORK_DIR}/readseek-android-arm64.tgz" "${TMP_WORK_DIR}/pi-readseek.tgz" --force --legacy-peer-deps > /dev/null 2>&1

echo "✅ Verification:"
"${BIN_DIR}/readseek" --version || echo "Readseek binary ready"

echo "🎉 Installed readseek and pi-readseek successfully!"
