#!/bin/bash
set -euo pipefail

if [[ $(uname -s) != Darwin || $(uname -m) != arm64 ]]; then
    echo 'macOS packaging requires native Apple Silicon macOS.' >&2
    exit 1
fi

route=${1:-onedir}
target=aarch64-apple-darwin
name=knowledgeagent-sidecar
case "$route" in
    onedir) ;;
    onefile) name="$name-$target" ;;
    *) echo 'Usage: build-macos.sh [onedir|onefile] [--version <semver>]' >&2; exit 2 ;;
esac

desktop=$(cd "$(dirname "$0")/.." && pwd)
workspace=$(cd "$desktop/.." && pwd)
backend="$desktop/../backend"
tauri="$desktop/../frontend/node_modules/.bin/tauri"
stage="$desktop/target/packaging"
p8_stage="$desktop/target/packaging"
variant=$route
config="$desktop/packaging/$route.json"
entry="$desktop/packaging/sidecar_entry.py"
version="0.3.0"
custom_version=0

if [[ $# -gt 1 ]]; then
    if [[ $# -eq 3 && "$route" == onedir && "$2" == --version ]]; then
        version="$3"
        custom_version=1
    else
        echo 'Usage: build-macos.sh [onedir|onefile] [--version <semver>]' >&2
        exit 2
    fi
fi
pyinstaller="$stage/pyinstaller"

UV_PROJECT_ENVIRONMENT="$stage/build-venv" uv run --project "$backend" \
    --python 3.12 --locked --no-dev --group packaging pyinstaller \
    "--$route" --console --target-arch arm64 --name "$name" --noconfirm \
    --recursive-copy-metadata pdfplumber --recursive-copy-metadata python-docx \
    --recursive-copy-metadata mcp --recursive-copy-metadata mcp-types \
    --hidden-import anyio._backends._asyncio \
    --collect-data jsonschema_specifications \
    --collect-data certifi \
    --paths "$backend" --distpath "$pyinstaller/$variant" \
    --workpath "$pyinstaller/work-$route" --specpath "$pyinstaller/spec-$route" \
    "$entry"

effective_config="$config"
if [[ "$custom_version" -eq 1 ]]; then
    mkdir -p "$p8_stage"
    effective_config="$p8_stage/onedir-$version.json"
    python3 -c '
import json, sys
base = json.load(open(sys.argv[1], encoding="utf-8"))
base["version"] = sys.argv[2]
json.dump(base, open(sys.argv[3], "w", encoding="utf-8"), indent=2)
' "$config" "$version" "$effective_config"
fi

cd "$desktop"
tauri_args=(build --ci --target "$target" --config "$effective_config")
if [[ "$route" == onefile ]]; then
    # Keep the original PyInstaller signature on externalBin for the local prototype.
    # Tauri's ad-hoc Hardened Runtime signature on it rejects the extracted dylibs.
    tauri_args+=(--no-sign)
fi
RUSTFLAGS="--remap-path-prefix=$HOME=~ --remap-path-prefix=$workspace=." \
    CARGO_TARGET_DIR="$desktop/target" "$tauri" "${tauri_args[@]}" -- --locked

output="$stage/apps/$variant/KnowledgeAgent.app"
if [[ "$custom_version" -eq 1 ]]; then
    output="$p8_stage/apps/onedir-$version/KnowledgeAgent.app"
fi
mkdir -p "$(dirname "$output")"
if [[ -e "$output" ]]; then
    rm -rf "$output"
fi
ditto "$desktop/target/$target/release/bundle/macos/KnowledgeAgent.app" "$output"
if [[ "$route" == onefile ]]; then
    cmp "$pyinstaller/$route/$name" "$output/Contents/MacOS/knowledgeagent-sidecar"
    codesign --force --sign - --options runtime "$output"
fi
codesign --verify --deep --strict "$output"
printf 'KnowledgeAgent app (%s, v%s): %s\n' "$variant" "$version" "$output"

if [[ "$route" == onedir && "$variant" == onedir ]]; then
    p8_app="$p8_stage/apps/onedir/KnowledgeAgent.app"
    if [[ "$custom_version" -eq 1 ]]; then
        p8_app="$output"
    elif [[ "$output" != "$p8_app" ]]; then
        mkdir -p "$(dirname "$p8_app")"
        rm -rf "$p8_app"
        ditto "$output" "$p8_app"
        codesign --verify --deep --strict "$p8_app"
    fi

    dmg_dir="$p8_stage/dmg"
    mkdir -p "$dmg_dir"
    dmg_out="$dmg_dir/KnowledgeAgent_${version}_aarch64-test.dmg"
    bundle_dmg="$desktop/target/$target/release/bundle/dmg/KnowledgeAgent_${version}_aarch64.dmg"
    staging_root=$(mktemp -d /private/tmp/knowledgeagent-dmg.XXXXXX)
    trap 'rm -rf "$staging_root"' EXIT
    ditto "$p8_app" "$staging_root/KnowledgeAgent.app"
    ln -s /Applications "$staging_root/Applications"
    mkdir -p "$staging_root/.background"
    cp "$desktop/packaging/dmg-background.png" "$staging_root/.background/dmg-background.png"
    cp "$desktop/packaging/INSTALL-GUIDE.txt" "$staging_root/安装引导（测试安装包）.txt"

    rm -f "$dmg_out"
    hdiutil create -volname "KnowledgeAgent" -srcfolder "$staging_root" \
        -ov -format UDZO "$dmg_out" >/dev/null
    hdiutil verify "$dmg_out" >/dev/null
    mkdir -p "$(dirname "$bundle_dmg")"
    cp -f "$dmg_out" "$bundle_dmg"
    printf 'KnowledgeAgent test DMG (v%s, ad-hoc signed): %s\n' "$version" "$dmg_out"
fi
