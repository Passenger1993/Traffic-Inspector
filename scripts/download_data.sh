#!/bin/bash
# scripts/download_data.sh
# Скрипт для загрузки и распаковки датасета BDD100K (семантическая сегментация) с Kaggle.
# Использует Kaggle API.

set -e  # Прерывать выполнение при ошибке

# Цвета для вывода
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Параметры по умолчанию
DEFAULT_DATA_DIR="data"
DATASET="solesensei/solesensei_bdd100k"
ARCHIVE_NAME="solesensei_bdd100k.zip"
REQUIRED_DIR_IN_ARCHIVE="bdd100k_seg"

# Функция вывода сообщения об ошибке и выхода
error() {
    echo -e "${RED}Ошибка: $1${NC}" >&2
    exit 1
}

info() {
    echo -e "${GREEN}[INFO] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[WARN] $1${NC}"
}

# Проверка наличия kaggle
if ! command -v kaggle &> /dev/null; then
    error "Kaggle CLI не установлен. Установите его: pip install kaggle"
fi

# Проверка авторизации Kaggle
if [ ! -f "$HOME/.kaggle/kaggle.json" ] && [ -z "$KAGGLE_USERNAME" ] && [ -z "$KAGGLE_KEY" ]; then
    error "Не найдена авторизация Kaggle. Создайте файл ~/.kaggle/kaggle.json или установите переменные окружения KAGGLE_USERNAME и KAGGLE_KEY"
fi

# Целевая директория для данных (первый аргумент, если указан)
DATA_DIR="${1:-$DEFAULT_DATA_DIR}"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

info "Загрузка датасета $DATASET в директорию $DATA_DIR"

# Проверяем, не скачан ли уже архив или папка
if [ -f "$ARCHIVE_NAME" ]; then
    warn "Архив $ARCHIVE_NAME уже существует. Использую его."
elif [ -d "$REQUIRED_DIR_IN_ARCHIVE" ]; then
    warn "Папка $REQUIRED_DIR_IN_ARCHIVE уже существует. Пропускаю загрузку."
    exit 0
else
    info "Скачиваю датасет с Kaggle..."
    kaggle datasets download -d "$DATASET" || error "Не удалось скачать датасет"
fi

# Проверяем, что архив существует после скачивания
if [ ! -f "$ARCHIVE_NAME" ]; then
    error "Архив $ARCHIVE_NAME не найден после загрузки"
fi

info "Распаковка только папки $REQUIRED_DIR_IN_ARCHIVE (это займёт некоторое время)..."
# Используем unzip с опцией -j (не создавать папки) не подходит, так как нам нужно сохранить структуру.
# Но мы извлечём только нужную папку.
unzip -q "$ARCHIVE_NAME" "$REQUIRED_DIR_IN_ARCHIVE/*" || error "Не удалось распаковать архив"

# Проверяем, что папка извлечена
if [ ! -d "$REQUIRED_DIR_IN_ARCHIVE" ]; then
    error "Папка $REQUIRED_DIR_IN_ARCHIVE не найдена после распаковки"
fi

# Удаляем архив для экономии места
info "Удаляю архив $ARCHIVE_NAME..."
rm "$ARCHIVE_NAME"

info "Датасет успешно загружен в $DATA_DIR/$REQUIRED_DIR_IN_ARCHIVE"
info "Структура:"
ls -la "$REQUIRED_DIR_IN_ARCHIVE"

info "Готово. Теперь укажите в конфиге пути к папкам images и labels внутри $DATA_DIR/$REQUIRED_DIR_IN_ARCHIVE."