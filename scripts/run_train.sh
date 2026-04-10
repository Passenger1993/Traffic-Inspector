#!/bin/bash
# scripts/run_train.sh
# Запуск обучения модели сегментации

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

CONFIG_FILE="${1:-configs/bdd100k.yaml}"
EXTRA_ARGS="${@:2}"

if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Ошибка: файл конфигурации $CONFIG_FILE не найден${NC}"
    exit 1
fi

echo -e "${GREEN}Запуск обучения с конфигом: $CONFIG_FILE${NC}"
echo "Дополнительные аргументы: $EXTRA_ARGS"

# Активация виртуального окружения
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
fi

python src/train.py --config "$CONFIG_FILE" $EXTRA_ARGS

echo -e "${GREEN}Обучение завершено${NC}"