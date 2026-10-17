#!/usr/bin/env bash
set -euo pipefail

ICON_SOURCE="${ICON_SOURCE:-assets/spektrafilm-icon.jpg}"
BUILD_TARGET_DIR="${BUILD_TARGET_DIR:-target/macos-app}"
DIST_DIR="${DIST_DIR:-dist/spektrafilm-macos}"
APP_NAME="${APP_NAME:-Spektrafilm}"
APP_BUNDLE="$DIST_DIR/$APP_NAME.app"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

cargo build --release -p spektrafilm-gui --target-dir "$BUILD_TARGET_DIR"
cargo build --release -p spektrafilm-cli --bin spektrafilm --target-dir "$BUILD_TARGET_DIR"
cargo build --release -p spektrafilm-cli --bin spektrafilm-f64 --features precision-f64 --target-dir "$BUILD_TARGET_DIR"

rm -rf "$DIST_DIR"
mkdir -p "$APP_BUNDLE/Contents/MacOS" "$APP_BUNDLE/Contents/Resources"

cp "$BUILD_TARGET_DIR/release/spektrafilm-gui" "$APP_BUNDLE/Contents/MacOS/spektrafilm-gui"
cp "$BUILD_TARGET_DIR/release/spektrafilm" "$APP_BUNDLE/Contents/MacOS/spektrafilm"
cp "$BUILD_TARGET_DIR/release/spektrafilm-f64" "$APP_BUNDLE/Contents/MacOS/spektrafilm-f64"
cp -R data "$APP_BUNDLE/Contents/Resources/data"
chmod +x "$APP_BUNDLE/Contents/MacOS/spektrafilm-gui" \
    "$APP_BUNDLE/Contents/MacOS/spektrafilm" \
    "$APP_BUNDLE/Contents/MacOS/spektrafilm-f64"

cat > "$APP_BUNDLE/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleDevelopmentRegion</key>
  <string>en</string>
  <key>CFBundleDisplayName</key>
  <string>Spektrafilm</string>
  <key>CFBundleExecutable</key>
  <string>spektrafilm-gui</string>
  <key>CFBundleIconFile</key>
  <string>spektrafilm.icns</string>
  <key>CFBundleIdentifier</key>
  <string>dev.spektrafilm.app</string>
  <key>CFBundleInfoDictionaryVersion</key>
  <string>6.0</string>
  <key>CFBundleName</key>
  <string>Spektrafilm</string>
  <key>CFBundlePackageType</key>
  <string>APPL</string>
  <key>CFBundleShortVersionString</key>
  <string>0.1.2</string>
  <key>CFBundleVersion</key>
  <string>0.1.2</string>
  <key>LSMinimumSystemVersion</key>
  <string>12.0</string>
  <key>NSHighResolutionCapable</key>
  <true/>
</dict>
</plist>
PLIST

if [[ -f "$ICON_SOURCE" ]]; then
    iconset="$BUILD_TARGET_DIR/spektrafilm.iconset"
    rm -rf "$iconset"
    mkdir -p "$iconset"
    sips -s format png -z 16 16 "$ICON_SOURCE" --out "$iconset/icon_16x16.png" >/dev/null
    sips -s format png -z 32 32 "$ICON_SOURCE" --out "$iconset/icon_16x16@2x.png" >/dev/null
    sips -s format png -z 32 32 "$ICON_SOURCE" --out "$iconset/icon_32x32.png" >/dev/null
    sips -s format png -z 64 64 "$ICON_SOURCE" --out "$iconset/icon_32x32@2x.png" >/dev/null
    sips -s format png -z 128 128 "$ICON_SOURCE" --out "$iconset/icon_128x128.png" >/dev/null
    sips -s format png -z 256 256 "$ICON_SOURCE" --out "$iconset/icon_128x128@2x.png" >/dev/null
    sips -s format png -z 256 256 "$ICON_SOURCE" --out "$iconset/icon_256x256.png" >/dev/null
    sips -s format png -z 512 512 "$ICON_SOURCE" --out "$iconset/icon_256x256@2x.png" >/dev/null
    sips -s format png -z 512 512 "$ICON_SOURCE" --out "$iconset/icon_512x512.png" >/dev/null
    sips -s format png -z 1024 1024 "$ICON_SOURCE" --out "$iconset/icon_512x512@2x.png" >/dev/null
    iconutil -c icns "$iconset" -o "$APP_BUNDLE/Contents/Resources/spektrafilm.icns"
fi

cat > "$DIST_DIR/README.txt" <<'README'
spektrafilm-rs macOS

Run Spektrafilm.app for the GUI.

Backends:
- macOS uses the WGSL/wgpu backend through Metal by default.
- CUDA is not available on modern macOS builds.

f64 export:
- spektrafilm-f64 is bundled inside Spektrafilm.app/Contents/MacOS.
- GPU full-resolution export is the default. Disable it under File export for CPU f64.
- The f64 CLI also powers native LUT export and bundle diagnostics.

Gatekeeper:
- This package is ad-hoc signed for local testing. It is not Apple-notarized.
- If macOS blocks it after download, right-click Spektrafilm.app and choose Open.
README

cp docs/upstream-sync.md "$DIST_DIR/REFERENCE_STATUS.md"
cp docs/parity-checklist.md "$DIST_DIR/VALIDATION_CHECKLIST.md"

if command -v codesign >/dev/null 2>&1; then
    codesign --force --deep --sign - "$APP_BUNDLE" >/dev/null
fi

zip_path="$DIST_DIR.zip"
rm -f "$zip_path"
(
    cd "$DIST_DIR/.."
    ditto -c -k --sequesterRsrc --keepParent "$(basename "$DIST_DIR")" "$(basename "$zip_path")"
)

echo "Packaged: $repo_root/$DIST_DIR"
echo "Zip:      $repo_root/$zip_path"
