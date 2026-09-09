import pandas as pd
from google_play_scraper import reviews, Sort

APP_ID = "id.dana"

result, continuation_token = reviews(
    APP_ID,
    lang="id",
    country="id",
    sort=Sort.NEWEST,
    count=50000
)

df_raw = pd.DataFrame(result)

df_raw.to_csv(
    "dana_raw_50000.csv",
    index=False,
    encoding="utf-8-sig"
)

print("Jumlah raw review:", len(df_raw))
print("Dataset berhasil disimpan.")