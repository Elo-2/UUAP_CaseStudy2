import pandas as pd

file_path = "data/US_youtube_trending_data.csv"

df = pd.read_csv(file_path)

print(df.head())

print("\nInformacije o datasetu:")
print(df.info())

print("\nDimenzije dataseta:")
print(df.shape)

print("\nNazivi kolona:")
print(df.columns.tolist())

print("\nNedostajuće vrijednosti:")
print(df.isnull().sum())

print("\nBroj duplikata:")
print(df.duplicated().sum())

print("\nOsnovna statistika numeričkih podataka:")
print(df.describe())

print("\nBroj zapisa po kategoriji:")
print(df["categoryId"].value_counts().sort_index())

df["publishedAt"] = pd.to_datetime(df["publishedAt"])

print("\nPeriod objavljivanja videa:")
print("Najstariji video:", df["publishedAt"].min())
print("Najnoviji video:", df["publishedAt"].max())

print("\nBroj videa po godinama:")

year_counts = df["publishedAt"].dt.year.value_counts().sort_index()

print(year_counts)

print("\nBroj ukupnih zapisa:", len(df))
print("Broj različitih videa:", df["video_id"].nunique())

video_counts = df["video_id"].value_counts()

print("\nBroj pojavljivanja videa:")
print(video_counts.describe())

print("\nNajčešće pojavljivani video:")
print(video_counts.head(10))

print("\nKorelacija između numeričkih varijabli:")

correlation = df[
    ["view_count", "likes", "dislikes", "comment_count"]
].corr()

print(correlation)

print("\nPercentili broja pregleda:")

print(df["view_count"].quantile([
    0.50,
    0.75,
    0.90,
    0.95,
    0.99,
    0.999
]))

category_mapping = {
    1: "Film & Animation",
    2: "Autos & Vehicles",
    10: "Music",
    15: "Pets & Animals",
    17: "Sports",
    19: "Travel & Events",
    20: "Gaming",
    22: "People & Blogs",
    23: "Comedy",
    24: "Entertainment",
    25: "News & Politics",
    26: "Howto & Style",
    27: "Education",
    28: "Science & Technology",
    29: "Nonprofits & Activism"
}

df["category_name"] = df["categoryId"].map(category_mapping)

print("\nKategorije:")
print(df["category_name"].value_counts())

df["trending_date"] = pd.to_datetime(df["trending_date"])

print("\nPeriod trending podataka:")
print("Prvi trending zapis:", df["trending_date"].min())
print("Zadnji trending zapis:", df["trending_date"].max())

first_trending = (
    df.groupby("video_id")["trending_date"]
    .min()
    .reset_index()
)

first_published = (
    df.groupby("video_id")["publishedAt"]
    .min()
    .reset_index()
)

video_timing = first_published.merge(
    first_trending,
    on="video_id"
)

video_timing["published_date"] = video_timing["publishedAt"].dt.normalize()
video_timing["trending_day"] = video_timing["trending_date"].dt.normalize()

video_timing["days_to_trending"] = (
    video_timing["trending_day"] - video_timing["published_date"]
).dt.days

print("\nVrijeme od objave do prvog trendinga:")
print(video_timing["days_to_trending"].describe())


print("\nBroj videa prema vremenu do prvog trendinga:")

trending_delay = (
    video_timing["days_to_trending"]
    .value_counts()
    .sort_index()
)

print(trending_delay)

import matplotlib.pyplot as plt
import seaborn as sns

# Broj trending zapisa po kategoriji
category_counts = (
    df["category_name"]
    .value_counts()
    .sort_values(ascending=False)
)

plt.figure(figsize=(12, 7))

sns.barplot(
    x=category_counts.values,
    y=category_counts.index
)

plt.title(
    "Broj trending zapisa prema YouTube kategoriji",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Broj trending zapisa")
plt.ylabel("Kategorija")

plt.tight_layout()

plt.savefig(
    "category_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# Vrijeme do prvog pojavljivanja u trendingu

delay_counts = (
    video_timing["days_to_trending"]
    .value_counts()
    .sort_index()
)

plt.figure(figsize=(12, 7))

sns.barplot(
    x=delay_counts.index,
    y=delay_counts.values
)

plt.title(
    "Vrijeme od objavljivanja do prvog pojavljivanja u trendingu",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Broj dana")
plt.ylabel("Broj videa")

plt.tight_layout()

plt.savefig(
    "trending_delay.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# Odnos pregleda i lajkova

plt.figure(figsize=(12, 7))

sns.scatterplot(
    data=df,
    x="view_count",
    y="likes",
    alpha=0.3
)

plt.title(
    "Odnos broja pregleda i lajkova",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Broj pregleda")
plt.ylabel("Broj lajkova")

plt.tight_layout()

plt.savefig(
    "views_vs_likes.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# Prosječan broj pregleda po kategoriji

category_views = (
    df.groupby("category_name")["view_count"]
    .mean()
    .sort_values(ascending=False)
)

print("\nProsječan broj pregleda po kategoriji:")
print(category_views)

# Medijan broja pregleda po kategoriji

category_views_median = (
    df.groupby("category_name")["view_count"]
    .median()
    .sort_values(ascending=False)
)

print("\nMedijan broja pregleda po kategoriji:")
print(category_views_median)

# Usporedba prosjeka i medijana pregleda po kategoriji

category_comparison = (
    df.groupby("category_name")["view_count"]
    .agg(["mean", "median", "count"])
    .sort_values("median", ascending=False)
)

print("\nUsporedba popularnosti kategorija:")
print(category_comparison)

# Uklanjanje potpuno dupliciranih redova

before_duplicates = len(df)

df = df.drop_duplicates().copy()

after_duplicates = len(df)

print("\nČišćenje duplikata:")
print("Broj redova prije:", before_duplicates)
print("Broj redova poslije:", after_duplicates)
print("Uklonjeno duplikata:", before_duplicates - after_duplicates)

# Popunjavanje nedostajućih opisa

missing_before = df["description"].isnull().sum()

df["description"] = df["description"].fillna("No description")

missing_after = df["description"].isnull().sum()

print("\nNedostajući opisi:")
print("Prije:", missing_before)
print("Poslije:", missing_after)

# Provjera negativnih vrijednosti

numeric_columns = [
    "view_count",
    "likes",
    "dislikes",
    "comment_count"
]

print("\nBroj negativnih vrijednosti:")

for column in numeric_columns:
    print(column, ":", (df[column] < 0).sum())


# Provjera vremenske logike

date_check = df["publishedAt"].dt.normalize() > df["trending_date"].dt.normalize()

print("\nBroj zapisa gdje je objava nakon trending datuma:")
print(date_check.sum())

# Popularnost kroz godine

df["year"] = df["publishedAt"].dt.year

yearly_popularity = (
    df.groupby("year")["view_count"]
    .agg(["count", "mean", "median"])
)

print("\nPopularnost trending videa po godinama:")
print(yearly_popularity)

# Grafikon medijana pregleda po godinama

plt.figure(figsize=(10, 6))

sns.lineplot(
    data=yearly_popularity,
    x=yearly_popularity.index,
    y="median",
    marker="o"
)

plt.title(
    "Medijan broja pregleda trending videa kroz godine",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Godina")
plt.ylabel("Medijan pregleda")

plt.xticks(yearly_popularity.index)

plt.tight_layout()

plt.savefig(
    "yearly_popularity.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# Izračun engagement rate-a

engagement_df = df[df["view_count"] > 0].copy()

engagement_df["engagement_rate"] = (
    (engagement_df["likes"] + engagement_df["comment_count"])
    / engagement_df["view_count"]
) * 100

print("\nEngagement rate:")
print(engagement_df["engagement_rate"].describe())

# Engagement rate po kategoriji

category_engagement = (
    engagement_df.groupby("category_name")["engagement_rate"]
    .agg(["mean", "median", "count"])
    .sort_values("median", ascending=False)
)

print("\nEngagement rate po kategoriji:")
print(category_engagement)

print("\nPercentili engagement rate-a:")

print(
    engagement_df["engagement_rate"].quantile([
        0.50,
        0.75,
        0.90,
        0.95,
        0.99,
        0.999
    ])
)

# Korelacija pregleda i engagement rate-a

views_engagement_correlation = engagement_df[
    ["view_count", "engagement_rate"]
].corr()

print("\nKorelacija između pregleda i engagement rate-a:")
print(views_engagement_correlation)

# Završna usporedba kategorija

category_final = (
    engagement_df.groupby("category_name")
    .agg(
        trending_records=("video_id", "count"),
        median_views=("view_count", "median"),
        median_engagement=("engagement_rate", "median")
    )
    .sort_values("median_engagement", ascending=False)
)

print("\nZavršna usporedba kategorija:")
print(category_final)

# Spremanje očišćenog dataseta

output_path = "data/youtube_trending_cleaned.csv"

df.to_csv(output_path, index=False)

print("\nOčišćeni dataset je spremljen:")
print(output_path)

# Vrijeme do prvog trendinga po kategoriji

video_categories = (
    df[["video_id", "category_name"]]
    .drop_duplicates("video_id")
)

timing_by_category = video_timing.merge(
    video_categories,
    on="video_id",
    how="left"
)

category_timing = (
    timing_by_category
    .groupby("category_name")["days_to_trending"]
    .agg(["mean", "median", "count"])
    .sort_values("median")
)

print("\nVrijeme do prvog trendinga po kategoriji:")
print(category_timing)

# Prosječno vrijeme do prvog trendinga po kategoriji

category_timing_plot = (
    category_timing
    .sort_values("mean", ascending=True)
)

plt.figure(figsize=(12, 7))

sns.barplot(
    data=category_timing_plot.reset_index(),
    x="mean",
    y="category_name"
)

plt.title(
    "Prosječno vrijeme do prvog pojavljivanja u trendingu po kategoriji",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Prosječan broj dana")
plt.ylabel("Kategorija")

plt.tight_layout()

plt.savefig(
    "category_trending_speed.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# Medijan engagement rate-a po kategoriji

category_engagement_plot = (
    category_engagement
    .sort_values("median", ascending=True)
)

plt.figure(figsize=(12, 7))

sns.barplot(
    data=category_engagement_plot.reset_index(),
    x="median",
    y="category_name"
)

plt.title(
    "Medijan engagement rate-a prema YouTube kategoriji",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Medijan engagement rate (%)")
plt.ylabel("Kategorija")

plt.tight_layout()

plt.savefig(
    "category_engagement.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# Medijan pregleda po kategoriji i godini

category_year_popularity = (
    df.groupby(["year", "category_name"])["view_count"]
    .median()
    .reset_index()
)

print("\nMedijan pregleda po kategoriji i godini:")
print(category_year_popularity)

# Kreiranje ciljne varijable za klasifikaciju

engagement_threshold = engagement_df["engagement_rate"].median()

engagement_df["high_engagement"] = (
    engagement_df["engagement_rate"] >= engagement_threshold
).astype(int)

print("\nPrag za visoki engagement:")
print(engagement_threshold)

print("\nRaspodjela ciljnih klasa:")
print(
    engagement_df["high_engagement"]
    .value_counts()
    .sort_index()
)

# Varijable za machine learning model

model_features = [
    "categoryId",
    "year",
    "comments_disabled",
    "ratings_disabled"
]

print("\nVarijable za machine learning model:")
print(model_features)

print("\nProvjera dostupnosti varijabli:")
print(engagement_df[model_features].info())

# Kreiranje dataset-a za machine learning

ml_df = engagement_df[
    [
        "video_id",
        "categoryId",
        "year",
        "comments_disabled",
        "ratings_disabled",
        "high_engagement"
    ]
].merge(
    video_timing[
        [
            "video_id",
            "days_to_trending"
        ]
    ],
    on="video_id",
    how="inner"
)

print("\nMachine learning dataset:")
print(ml_df.info())

print("\nPrvih 5 redova ML dataset-a:")
print(ml_df.head())

# Priprema podataka za machine learning

from sklearn.model_selection import train_test_split

X = ml_df[
    [
        "categoryId",
        "year",
        "comments_disabled",
        "ratings_disabled",
        "days_to_trending"
    ]
]

y = ml_df["high_engagement"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nPodjela podataka za machine learning:")
print("Trening skup:", X_train.shape)
print("Test skup:", X_test.shape)

print("\nRaspodjela klasa u trening skupu:")
print(y_train.value_counts())

print("\nRaspodjela klasa u test skupu:")
print(y_test.value_counts())

# Random Forest model

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("\nRandom Forest model je uspješno treniran.")

# Evaluacija Random Forest modela

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nRezultati Random Forest modela:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")

# Važnost varijabli u Random Forest modelu

feature_importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": model.feature_importances_
}).sort_values(
    "importance",
    ascending=False
)

print("\nVažnost varijabli:")
print(feature_importance)

# Confusion matrix Random Forest modela

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Low engagement", "High engagement"]
)

disp.plot()

plt.title(
    "Confusion Matrix - Random Forest",
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# Grafikon važnosti varijabli

feature_importance_plot = feature_importance.sort_values(
    "importance",
    ascending=True
)

plt.figure(figsize=(10, 6))

sns.barplot(
    data=feature_importance_plot,
    x="importance",
    y="feature"
)

plt.title(
    "Važnost varijabli u Random Forest modelu",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Važnost")
plt.ylabel("Varijabla")

plt.tight_layout()

plt.savefig(
    "feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# Logistic Regression model

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

logistic_model = Pipeline([
    ("scaler", StandardScaler()),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    )
])

logistic_model.fit(X_train, y_train)

logistic_pred = logistic_model.predict(X_test)

logistic_accuracy = accuracy_score(y_test, logistic_pred)
logistic_precision = precision_score(y_test, logistic_pred)
logistic_recall = recall_score(y_test, logistic_pred)
logistic_f1 = f1_score(y_test, logistic_pred)

print("\nRezultati Logistic Regression modela:")
print(f"Accuracy:  {logistic_accuracy:.4f}")
print(f"Precision: {logistic_precision:.4f}")
print(f"Recall:    {logistic_recall:.4f}")
print(f"F1-score:  {logistic_f1:.4f}")

# Konačna usporedba machine learning modela

model_comparison = pd.DataFrame({
    "Model": [
        "Random Forest",
        "Logistic Regression"
    ],
    "Accuracy": [
        accuracy,
        logistic_accuracy
    ],
    "Precision": [
        precision,
        logistic_precision
    ],
    "Recall": [
        recall,
        logistic_recall
    ],
    "F1-score": [
        f1,
        logistic_f1
    ]
})

print("\nKonačna usporedba modela:")
print(model_comparison.to_string(index=False))

# Konačne statistike za poster

poster_statistics = {
    "Početni broj zapisa": len(pd.read_csv(file_path)),
    "Broj zapisa nakon čišćenja": len(df),
    "Broj različitih videa": df["video_id"].nunique(),
    "Broj varijabli": len(df.columns),
    "Medijan vremena do trendinga (dani)": video_timing["days_to_trending"].median(),
    "Najčešća kategorija": df["category_name"].value_counts().idxmax(),
    "Broj zapisa najčešće kategorije": df["category_name"].value_counts().max(),
    "Najveći medijan engagementa": category_engagement["median"].idxmax(),
    "Vrijednost najvećeg medijana engagementa (%)": category_engagement["median"].max(),
    "Najveći medijan pregleda": category_views_median.idxmax(),
    "Vrijednost najvećeg medijana pregleda": category_views_median.max(),
    "Korelacija pregleda i lajkova": df[
        ["view_count", "likes"]
    ].corr().loc["view_count", "likes"],
    "Korelacija pregleda i engagementa": views_engagement_correlation.loc[
        "view_count", "engagement_rate"
    ],
    "Random Forest Accuracy": accuracy,
    "Random Forest F1-score": f1,
    "Random Forest Recall": recall,
    "Najvažnija ML varijabla": feature_importance.iloc[0]["feature"],
    "Važnost najvažnije varijable (%)": feature_importance.iloc[0]["importance"] * 100
}

print("\n" + "=" * 60)
print("KONAČNE STATISTIKE ZA POSTER")
print("=" * 60)

for key, value in poster_statistics.items():
    print(f"{key}: {value}")

# Spremanje ključnih rezultata za poster

with open("poster_results.txt", "w", encoding="utf-8") as file:
    file.write("YOUTUBE TRENDING ANALYSIS - KLJUČNI REZULTATI\n")
    file.write("=" * 60 + "\n\n")

    file.write("DATASET\n")
    file.write(f"Početni broj zapisa: {len(pd.read_csv(file_path))}\n")
    file.write(f"Broj zapisa nakon čišćenja: {len(df)}\n")
    file.write(f"Broj različitih videa: {df['video_id'].nunique()}\n")
    file.write("Broj originalnih varijabli: 16\n")
    file.write("Period objave: 2020-2024\n")
    file.write("Period trending podataka: 2020-2024\n\n")

    file.write("GLAVNI EDA REZULTATI\n")
    file.write(f"Medijan vremena do prvog trendinga: "
               f"{video_timing['days_to_trending'].median():.0f} dan\n")
    file.write(f"Najčešća kategorija: "
               f"{df['category_name'].value_counts().idxmax()}\n")
    file.write(f"Broj zapisa najčešće kategorije: "
               f"{df['category_name'].value_counts().max()}\n")
    file.write(f"Najveći medijan engagementa: "
               f"{category_engagement['median'].idxmax()}\n")
    file.write(f"Medijan engagementa: "
               f"{category_engagement['median'].max():.2f}%\n")
    file.write(f"Najveći medijan pregleda: "
               f"{category_views_median.idxmax()}\n")
    file.write(f"Medijan pregleda: "
               f"{category_views_median.max():,.0f}\n")
    file.write(f"Korelacija pregleda i lajkova: "
               f"{df[['view_count', 'likes']].corr().loc['view_count', 'likes']:.3f}\n")
    file.write(f"Korelacija pregleda i engagementa: "
               f"{views_engagement_correlation.loc['view_count', 'engagement_rate']:.3f}\n\n")

    file.write("MACHINE LEARNING\n")
    file.write("Glavni model: Random Forest\n")
    file.write(f"Accuracy: {accuracy:.2%}\n")
    file.write(f"Precision: {precision:.2%}\n")
    file.write(f"Recall: {recall:.2%}\n")
    file.write(f"F1-score: {f1:.2%}\n")
    file.write(f"Najvažnija varijabla: "
               f"{feature_importance.iloc[0]['feature']}\n")
    file.write(f"Važnost najvažnije varijable: "
               f"{feature_importance.iloc[0]['importance']:.2%}\n\n")

    file.write("ISTRAŽIVAČKO PITANJE\n")
    file.write(
        "Koje karakteristike YouTube videa najviše utiču na njihovu "
        "popularnost i engagement, te koliko uspješno možemo predvidjeti "
        "visoki engagement?\n\n"
    )

    file.write("HIPOTEZA\n")
    file.write(
        "H1: Kategorija videa značajno je povezana s nivoom engagementa "
        "i predstavlja važan prediktor visokog engagementa.\n"
    )

print("\nKljučni rezultati spremljeni u poster_results.txt")