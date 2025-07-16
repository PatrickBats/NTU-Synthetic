import settings
import json
import os

def save_settings():
    # 确保输出目录存在
    if not os.path.exists(settings.OUTPUT_FOLDER):
        os.makedirs(settings.OUTPUT_FOLDER)

    # 将设置参数保存为JSON文件
    settings_dict = {
        "TEST_MODE": settings.TEST_MODE,
        "MODEL": settings.MODEL,
        "QUANTILE": settings.QUANTILE,
        "SEG_THRESHOLD": settings.SEG_THRESHOLD,
        "SCORE_THRESHOLD": settings.SCORE_THRESHOLD,
        "TOPN": settings.TOPN,
        "PARALLEL": settings.PARALLEL,
        "CATAGORIES": settings.CATAGORIES,
        "FEATURE_NAMES": settings.FEATURE_NAMES,
        "MODEL_PARALLEL": settings.MODEL_PARALLEL,
        "WORKERS": settings.WORKERS,
        "BATCH_SIZE": settings.BATCH_SIZE,
        "TALLY_BATCH_SIZE": settings.TALLY_BATCH_SIZE,
        "INDEX_FILE": settings.INDEX_FILE
    }

    with open(os.path.join(settings.OUTPUT_FOLDER, "settings.json"), 'w') as f:
        json.dump(settings_dict, f, indent=4)
    print("Settings saved to settings.json")