from huggingface_hub import login, HfApi, upload_folder

# Вход (потребуется токен)
login(token="TOKEN")

# Создание репозитория (если ещё не создан)
api = HfApi()
api.upload_large_folder(
    repo_id="Alex-Watchman/bdd100k-vehicle-segmentation",
    repo_type="dataset",
    folder_path="data/bdd100k_seg",
    num_workers=2,          # количество потоков (можно увеличить/уменьшить)
)

print("Датасет успешно загружен!")