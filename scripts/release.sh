#!/usr/bin/env bash

# リリース自動化スクリプト
# このスクリプトはリリースプロセスを自動化します:
# 1. Conventional Commitsを解析して次のバージョンを提案
# 2. pyproject.tomlと__init__.pyのバージョンを更新
# 3. gitコミットとタグを作成
# 4. タグをプッシュしてGitHub Actionsリリースワークフローをトリガー

set -euo pipefail

# 出力用カラーコード
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # カラーなし

# バージョン情報を含むファイル
readonly PYPROJECT_FILE="pyproject.toml"
readonly INIT_FILE="src/vtt2minutes/__init__.py"

# カラー付きメッセージを出力する関数
info() {
    echo -e "${BLUE}ℹ${NC} $*"
}

success() {
    echo -e "${GREEN}✓${NC} $*"
}

warning() {
    echo -e "${YELLOW}⚠${NC} $*"
}

error() {
    echo -e "${RED}✗${NC} $*" >&2
}

# pyproject.tomlから現在のバージョンを取得する関数
get_current_version() {
    grep '^version = ' "$PYPROJECT_FILE" | sed 's/version = "\(.*\)"/\1/'
}

# コミットを解析して次のバージョンを提案する関数
suggest_next_version() {
    local current_version="$1"
    local major minor patch

    # 現在のバージョンを解析 (semver形式を想定: X.Y.Z)
    if [[ $current_version =~ ^([0-9]+)\.([0-9]+)\.([0-9]+)$ ]]; then
        major="${BASH_REMATCH[1]}"
        minor="${BASH_REMATCH[2]}"
        patch="${BASH_REMATCH[3]}"
    else
        error "無効なバージョン形式: $current_version"
        exit 1
    fi

    # 最後のタグ以降のコミットを取得
    local last_tag
    last_tag=$(git describe --tags --abbrev=0 2>/dev/null || echo "")

    local commits
    if [[ -n "$last_tag" ]]; then
        commits=$(git log "${last_tag}..HEAD" --pretty=format:"%s" 2>/dev/null || echo "")
    else
        commits=$(git log --pretty=format:"%s" 2>/dev/null || echo "")
    fi

    # バージョンアップの種類を判定
    local has_breaking=false
    local has_feat=false
    local has_fix=false

    while IFS= read -r commit; do
        if [[ "$commit" =~ BREAKING[[:space:]]CHANGE || "$commit" =~ ^[a-z]+(\(.+\))?!: ]]; then
            has_breaking=true
        elif [[ "$commit" =~ ^feat ]]; then
            has_feat=true
        elif [[ "$commit" =~ ^fix ]]; then
            has_fix=true
        fi
    done <<< "$commits"

    # Conventional Commitsに基づいて次のバージョンを計算
    if [[ "$has_breaking" == true ]]; then
        echo "$((major + 1)).0.0"
    elif [[ "$has_feat" == true ]]; then
        echo "${major}.$((minor + 1)).0"
    elif [[ "$has_fix" == true ]]; then
        echo "${major}.${minor}.$((patch + 1))"
    else
        # デフォルトではパッチバージョンをアップ
        echo "${major}.${minor}.$((patch + 1))"
    fi
}

# ファイル内のバージョンを更新する関数
update_version() {
    local new_version="$1"

    info "バージョンを $new_version に更新中..."

    # pyproject.tomlを更新
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        sed -i '' "s/^version = \".*\"/version = \"${new_version}\"/" "$PYPROJECT_FILE"
    else
        # Linux
        sed -i "s/^version = \".*\"/version = \"${new_version}\"/" "$PYPROJECT_FILE"
    fi

    # __init__.pyを更新
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        sed -i '' "s/^__version__ = \".*\"/__version__ = \"${new_version}\"/" "$INIT_FILE"
    else
        # Linux
        sed -i "s/^__version__ = \".*\"/__version__ = \"${new_version}\"/" "$INIT_FILE"
    fi

    success "$PYPROJECT_FILE と $INIT_FILE のバージョンを更新しました"

    # uv.lockを更新
    info "uv.lock を更新中..."
    uv lock
    success "uv.lock を更新しました"
}

# リリースコミットとタグを作成する関数
create_release() {
    local version="$1"
    local tag="v${version}"

    info "リリースコミットとタグを作成中..."

    # バージョンファイルとロックファイルをステージング
    git add "$PYPROJECT_FILE" "$INIT_FILE" uv.lock

    # リリースコミットを作成
    git commit -m "chore(release): prepare for ${version}"

    # 注釈付きタグを作成
    git tag -a "$tag" -m "Release ${version}"

    success "コミットとタグを作成しました: $tag"
}

# タグをリモートにプッシュする関数
push_release() {
    local version="$1"
    local tag="v${version}"

    info "タグをリモートリポジトリにプッシュ中..."

    # コミットとタグをプッシュ
    git push origin main
    git push origin "$tag"

    success "タグ $tag をリモートにプッシュしました"
    info "GitHub Actionsが自動的にリリースを作成します"
}

# メイン関数
main() {
    info "リリースプロセスを開始します..."

    # gitリポジトリ内にいるかチェック
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        error "gitリポジトリ内にいません"
        exit 1
    fi

    # 未コミットの変更がないかチェック
    if ! git diff-index --quiet HEAD --; then
        error "未コミットの変更があります。先にコミットまたはスタッシュしてください。"
        exit 1
    fi

    # 現在のバージョンを取得
    local current_version
    current_version=$(get_current_version)
    info "現在のバージョン: $current_version"

    # 次のバージョンを提案
    local suggested_version
    suggested_version=$(suggest_next_version "$current_version")
    info "提案される次のバージョン: $suggested_version"

    # ユーザーにバージョンの確認または変更を求める
    echo ""
    read -rp "リリースするバージョンを入力してください (デフォルト: $suggested_version): " user_version
    local new_version="${user_version:-$suggested_version}"

    # バージョン形式を検証
    if ! [[ "$new_version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        error "無効なバージョン形式です。semver形式 (X.Y.Z) を使用してください"
        exit 1
    fi

    # リリースの確認
    echo ""
    warning "以下の処理を実行します:"
    echo "  1. バージョンを $new_version に更新"
    echo "  2. コミットとタグ v$new_version を作成"
    echo "  3. リモートリポジトリにプッシュ"
    echo "  4. GitHub Actionsをトリガーしてリリースを作成"
    echo ""
    read -rp "続行しますか? (y/N): " confirm

    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        warning "リリースをキャンセルしました"
        exit 0
    fi

    # リリース手順を実行
    update_version "$new_version"
    create_release "$new_version"
    push_release "$new_version"

    echo ""
    success "リリースプロセスが正常に完了しました!"
    info "リリースを確認するには https://github.com/kiririmode/vtt2minutes/releases にアクセスしてください"
}

# メイン関数を実行
main "$@"
