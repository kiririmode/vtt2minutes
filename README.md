# VTT2Minutes

Microsoft Teams の VTT 形式議事録ファイルから、Amazon Bedrock を活用して AI による議事録を自動生成する Python ツールです。

## 特徴

- **VTT ファイル解析**: Microsoft Teams の WebVTT 形式トランスクリプトを分析
- **高度な前処理**: フィラーワード除去、単語置換、ノイズ軽減、重複除去による文字起こし品質の向上
- **日本語特化**: 日本語のフィラーワードと句読点処理に最適化
- **AI 議事録生成**: Amazon Bedrock を使用した知的でコンテキストを理解した議事録生成
- **発言者識別**: 各発言者の貢献を適切に識別・分類
- **中間ファイル出力**: 前処理済みトランスクリプトを構造化された Markdown 形式で出力

## インストール

### オプション 1: スタンドアロンバイナリ（推奨）

[リリースページ](https://github.com/kiririmode/vtt2minutes/releases) から事前ビルド済みバイナリをダウンロード：

**Linux (x86_64):**
```bash
# ダウンロードと展開
wget https://github.com/kiririmode/vtt2minutes/releases/latest/download/vtt2minutes-linux-x86_64.tar.gz
tar -xzf vtt2minutes-linux-x86_64.tar.gz

# 直接実行
./vtt2minutes --help
```

**Windows (x86_64):**
1. リリースから `vtt2minutes-windows-x86_64.zip` をダウンロード
2. zip ファイルを展開
3. コマンドプロンプトまたは PowerShell で `vtt2minutes.exe --help` を実行

### オプション 2: Python インストール

**要件:**
- Python 3.12 以上
- uv（推奨パッケージマネージャー）
- Amazon Bedrock アクセス権限を持つ AWS アカウント
- 設定済み AWS 認証情報

**セットアップ:**
```bash
# リポジトリのクローン
git clone https://github.com/kiririmode/vtt2minutes.git
cd vtt2minutes

# 依存関係のインストール
uv sync

# 実行
uv run vtt2minutes --help
```

## AWS 設定

### 必要な権限

AWS IAM ユーザーまたはロールに以下の権限が必要です：

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel",
                "bedrock:ListFoundationModels"
            ],
            "Resource": "*"
        }
    ]
}
```

### 認証情報の設定

#### オプション 1: 環境変数

```bash
export AWS_ACCESS_KEY_ID=your-access-key-id
export AWS_SECRET_ACCESS_KEY=your-secret-access-key
export AWS_DEFAULT_REGION=ap-northeast-1
export AWS_SESSION_TOKEN=your-session-token  # 一時的な認証情報を使用する場合のみ
```

#### オプション 2: AWS CLI プロファイル

```bash
aws configure set aws_access_key_id your-access-key-id
aws configure set aws_secret_access_key your-secret-access-key
aws configure set region ap-northeast-1
```

## 使用方法

### 基本的な使用方法

```bash
# VTT ファイルから議事録を生成
vtt2minutes meeting.vtt

# カスタムタイトルと出力ファイルを指定
vtt2minutes meeting.vtt -o minutes.md -t "プロジェクト企画会議"

# 特定の Bedrock モデルを使用（Claude Sonnet 4）
vtt2minutes meeting.vtt --bedrock-model anthropic.claude-sonnet-4-20250514

# または推論プロファイルを使用
vtt2minutes meeting.vtt --bedrock-inference-profile-id apac-claude-sonnet-4

# 詳細出力と統計情報を表示
vtt2minutes meeting.vtt --verbose --stats
```

### 高度なオプション

```bash
# 前処理をスキップ
vtt2minutes meeting.vtt --no-preprocessing

# カスタムフィラーワードファイルを使用
vtt2minutes meeting.vtt --filter-words-file custom_filler_words.txt

# カスタム単語置換ルールを使用
vtt2minutes meeting.vtt --replacement-rules-file custom_rules.txt

# 中間ファイルを保存
vtt2minutes meeting.vtt --intermediate-file processed_transcript.md

# カスタムプロンプトテンプレートを使用
vtt2minutes meeting.vtt --prompt-template custom_template.txt
```

## 設定ファイル

### 単語置換ルール (`replacement_rules.txt`)

技術用語や略語を標準化するための単語置換をカスタマイズ：

```text
# 技術用語の統一
ベッドロック -> Bedrock
ラムダ -> Lambda
エス3 -> S3
イーシー2 -> EC2

# 会社・製品名の統一
アマゾン -> Amazon
グーグル -> Google
マイクロソフト -> Microsoft

# 一般的な略語展開
API -> アプリケーションプログラミングインターフェース
UI -> ユーザーインターフェース
DB -> データベース
```

### フィラーワード

デフォルトで除去される日本語フィラーワード：

#### 日本語
- えー, あー, うー, そのー, なんか, まあ, ちょっと
- えっと, あのー, そうですね, はい, ええ, うん

#### 転写アーティファクト
- [音声が途切れました], [雑音], [不明瞭], [咳], [笑い]

## カスタムフィラーワードファイル

デフォルトのフィラーワードを上書きするカスタムフィラーワードファイルを作成できます：

```txt
# カスタムフィラーワード
# # で始まる行はコメントとして無視されます
# 空行も無視されます

# カスタム日本語フィラーワード
えっと
そのー
まぁ

# 追加のカスタムフィラーワード
まじで
やっぱり

# カスタム転写アーティファクト
[microphone feedback]
[phone ringing]
```

## Bedrock モデル設定

### サポートされているモデル

- **Claude Sonnet 4** (最新・最高品質): `anthropic.claude-sonnet-4-20250514`
- **Claude 3.5 Sonnet** (推奨): `anthropic.claude-3-5-sonnet-20241022-v2:0`
- **Claude 3 Haiku**: `anthropic.claude-3-haiku-20240307-v1:0`
- **Claude 3 Sonnet**: `anthropic.claude-3-sonnet-20240229-v1:0`

### 推奨設定

```bash
# 最高品質な議事録生成（最新）
vtt2minutes meeting.vtt --bedrock-model anthropic.claude-sonnet-4-20250514

# 高品質な議事録生成（安定版推奨）
vtt2minutes meeting.vtt --bedrock-model anthropic.claude-3-5-sonnet-20241022-v2:0

# 高速で費用効率的な処理
vtt2minutes meeting.vtt --bedrock-model anthropic.claude-3-haiku-20240307-v1:0
```

### Bedrock Cross-Region Inference（推論プロファイル）

APAC リージョンで利用可能な推論プロファイルを使用してコストと可用性を最適化：

```bash
# Claude Sonnet 4 推論プロファイル（最新・最高品質）
vtt2minutes meeting.vtt --bedrock-inference-profile-id apac-claude-sonnet-4

# Claude 3.5 Sonnet 推論プロファイル（推奨）
vtt2minutes meeting.vtt --bedrock-inference-profile-id apac-claude-3-5-sonnet

# Claude 3 Haiku 推論プロファイル（高速・費用効率）
vtt2minutes meeting.vtt --bedrock-inference-profile-id apac-claude-3-haiku

# 特定のリージョンを指定
vtt2minutes meeting.vtt --bedrock-region ap-northeast-1 --bedrock-inference-profile-id apac-claude-3-5-sonnet
```

**推論プロファイルの利点:**
- **コスト最適化**: リージョン間でのコスト効率的なルーティング
- **可用性向上**: 複数リージョンでの冗長性確保
- **レイテンシー改善**: 最適なリージョンへの自動ルーティング

**注意**: `--bedrock-model` と `--bedrock-inference-profile-id` は相互排他的です。どちらか一方のみを指定してください。

## プロンプトテンプレート

### デフォルトテンプレート

デフォルトで `prompt_templates/default.txt` が使用され、以下の構造で議事録を生成：

- 議題
- 参加者
- 議論のまとめ
- 決定事項
- 議事
- 今後のアクション
- 特記事項

### カスタムテンプレート

カスタムプロンプトテンプレートを作成してより詳細な制御が可能：

```text
以下の要件に従って議事録を作成してください。

タイトル: {title}

前処理済み会議記録:
{markdown_content}

要件:
1. 重要な決定事項を明確に抽出する
2. アクションアイテムと担当者を特定する
3. 次回会議までのタスクを整理する
```

プレースホルダー:
- `{title}`: 会議タイトル
- `{markdown_content}`: 前処理済みトランスクリプト内容

## 出力例

生成される議事録のサンプル：

```markdown
# プロジェクト企画会議

## 会議概要
- 日時: 2024年1月15日
- 参加者: 田中氏、佐藤氏、山田氏
- 目的: 新プロジェクトの企画検討

## 主要な議題
1. プロジェクトスコープの定義
2. リソース配分の検討
3. スケジュール策定

## 決定事項
- プロジェクト開始日を2024年2月1日に決定
- 初期予算として500万円を承認
- チームリーダーに田中氏を任命

## アクションアイテム
- [ ] 詳細な要件定義書の作成（田中氏、1月31日まで）
- [ ] 開発チームの採用活動開始（佐藤氏、1月25日まで）
- [ ] プロジェクト管理ツールの選定（山田氏、1月20日まで）

## 次回までの課題
- 技術スタックの最終決定
- 外部パートナーとの契約締結
- プロジェクト管理体制の構築

## 詳細な議論内容
### プロジェクトスコープについて
田中氏より、プロジェクトの目標と範囲について詳細な説明がありました。
特に、ユーザビリティと拡張性を重視する方針が確認されました。

### リソース配分について
佐藤氏から人員配置の提案があり、全員で検討した結果、
フロントエンド2名、バックエンド3名の体制で進めることになりました。
```

## 開発・貢献

### 環境セットアップ

```bash
# リポジトリのクローン
git clone https://github.com/kiririmode/vtt2minutes.git
cd vtt2minutes

# 開発環境のセットアップ
./scripts/setup-hooks.sh
uv sync --extra dev

# コード品質チェック
uv run ruff format .  # フォーマット
uv run ruff check .   # リンティング
uv run pyright        # 型チェック

# テスト実行
uv run pytest         # 全テスト実行
uv run pytest -v      # 詳細出力
uv run pytest --cov   # カバレッジ付き

# 特定のテスト実行
uv run pytest tests/test_parser.py
```

### プリコミットフック

品質チェックのためのプリコミットフックが自動的にインストールされ、コミット前に以下をチェック：

- フォーマット（ruff）
- リンティング（ruff）
- 型チェック（pyright）
- テスト（pytest）

一時的にバイパスするには（推奨しません）：
```bash
git commit --no-verify
```

### ビルドとテスト

```bash
# バイナリビルド
./scripts/build-binary.sh --platform linux
./scripts/build-binary.sh --platform windows

# テスト環境での実行
PYTEST_DISABLE_PLUGIN_AUTOLOAD="" uv run pytest  # プラグインエラー回避
```

## トラブルシューティング

### よくある問題

**1. AWS 認証エラー**
```
Error: Unable to locate credentials
```
→ AWS 認証情報が正しく設定されているか確認してください

**2. Bedrock モデルアクセスエラー**
```
Error: Model access denied
```
→ AWS コンソールで Bedrock モデルへのアクセス許可を確認してください

**3. VTT ファイル解析エラー**
```
Error: Invalid VTT format
```
→ VTT ファイルが Microsoft Teams の正しい形式かどうか確認してください

**4. プロンプトテンプレートエラー**
```
Error: Failed to read prompt template file
```
→ テンプレートファイルのパスと権限を確認してください

### パフォーマンスの最適化

- **大きなファイル**: 長時間の会議録の場合は Claude 3 Haiku を使用して処理時間を短縮
- **高品質出力**: 重要な会議では Claude 3.5 Sonnet を使用して最高品質の議事録を生成
- **前処理**: 複雑な前処理が不要な場合は `--no-preprocessing` を使用

## ライセンス

このプロジェクトは MIT ライセンスの下で公開されています。詳細は [LICENSE](LICENSE) ファイルを参照してください。

## 貢献

プルリクエストやイシューは歓迎します。貢献ガイドラインについては、開発セクションを参照してください。

## サポート

- **バグレポート**: [GitHub Issues](https://github.com/kiririmode/vtt2minutes/issues)
- **機能リクエスト**: [GitHub Issues](https://github.com/kiririmode/vtt2minutes/issues)
- **ディスカッション**: [GitHub Discussions](https://github.com/kiririmode/vtt2minutes/discussions)