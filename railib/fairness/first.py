import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from pycm import ConfusionMatrix
from IPython.display import display, Markdown


feture_to_title_dict = {
    "occupation": "Ground-Truth Occupation",
    "prediction": "Prediction",
    "gender": "Gender",
}


def sampled(df, display_list=["occupation", "prediction", "gender"]):
    for _, row in df.sample(10).iterrows():
        for feature in display_list:
            display(Markdown(f"### {feture_to_title_dict[feature]}: " + row[feature]))
        display(Markdown(row["bio"]))
        display(Markdown("----"))


def train(train_df):
    # feature engineering: Bag of Words
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(train_df["bio"])

    # model: Logistic Regression
    # might take a minute or two to run
    model = SGDClassifier().fit(X_train_counts, train_df["occupation"])
    return count_vect, model


def accuracy_score(df, model, count_vect):
    df_counts = count_vect.transform(df["bio"])
    return model.score(df_counts, df["occupation"])


def predict(bios):
    X_counts = count_vect.transform(bios)
    return model.predict(X_counts)


def classification_metrics(df, label_col, prediction_col):
    cm = ConfusionMatrix(df[label_col].values, df[prediction_col].values)
    acceptance_rate = df[prediction_col].value_counts(normalize=True)

    df = pd.DataFrame({"ar": acceptance_rate, "fnr": cm.FNR, "fpr": cm.FPR})
    df.index.name = "occupation"
    return df


def unfairness_metrics_df(test_df):
    # seperate the test dataset into two gendered datasets
    male_test_df = test_df[test_df["gender"] == "M"]
    female_test_df = test_df[test_df["gender"] == "F"]

    # create the list of occupations
    occupation = test_df["occupation"].unique().tolist()

    # calculate the classification metrics per occuation
    # on the female and male subset of the test dataset
    male_metrics_df = classification_metrics(male_test_df, "occupation", "prediction")
    female_metrics_df = classification_metrics(
        female_test_df, "occupation", "prediction"
    )

    # merge the metr
    unfairness_metrics_df = pd.merge(
        male_metrics_df,
        female_metrics_df,
        left_index=True,
        right_index=True,
        suffixes=("_male", "_female"),
    )

    # sot columns
    unfairness_metrics_df = unfairness_metrics_df[
        ["ar_female", "ar_male", "fnr_female", "fnr_male", "fpr_female", "fpr_male"]
    ]
    return unfairness_metrics_df


def plt_unfairness_metrics(test_df):
    male_test_df = test_df[test_df["gender"] == "M"]
    female_test_df = test_df[test_df["gender"] == "F"]

    male_metrics_df = classification_metrics(male_test_df, "occupation", "prediction")
    female_metrics_df = classification_metrics(
        female_test_df, "occupation", "prediction"
    )
    unfairness_metrics_wide_df = pd.concat(
        [
            female_metrics_df.assign(gender="female"),
            male_metrics_df.assign(gender="male"),
        ]
    ).reset_index()

    # Make the PairGrid
    g = sns.PairGrid(
        unfairness_metrics_wide_df,  # .sort_values('ar', ascending=False),
        x_vars=["ar", "fnr", "fpr"],
        y_vars=["occupation"],
        hue="gender",
        height=10,
        aspect=0.25,
    )

    # Draw a dot plot using the stripplot function
    g.map(
        sns.stripplot,
        size=15,
        orient="h",
        jitter=False,
        alpha=0.6,
        linewidth=1,
        edgecolor="w",
    )

    # # Use the same x axis limits on all columns and add better labels
    g.set(xlim=(-0.05, 0.7), xlabel="Value", ylabel="")

    # Use semantically meaningful titles for the columns
    titles = ["Acceptance Rate", "False Negative Rate", "False Positive Rate"]

    for ax, title in zip(g.axes.flat, titles):

        # Set a different title for each axes
        ax.set(title=title)

        # Make the grid horizontal instead of vertical
        ax.xaxis.grid(False)
        ax.yaxis.grid(True)

    sns.despine(left=True, bottom=True)
    plt.legend(loc="center right", bbox_to_anchor=(1.6, 0.5))


def unfairness_metrics_df_gap(unfairness_metrics_df):
    # calculate the gap (difference) of the classification metrics
    # between female indivdiuals and male individuals
    # per occupation
    for metric in ("ar", "fnr", "fpr"):
        unfairness_metrics_df[f"{metric}_gap"] = (
            unfairness_metrics_df[f"{metric}_female"]
            - unfairness_metrics_df[f"{metric}_male"]
        )

    unfairness_metrics_df = unfairness_metrics_df.round(2)

    # order columns
    unfairness_metrics_df = unfairness_metrics_df[
        [
            "ar_female",
            "ar_male",
            "ar_gap",
            "fnr_female",
            "fnr_male",
            "fnr_gap",
            "fpr_female",
            "fpr_male",
            "fpr_gap",
        ]
    ]

    return unfairness_metrics_df
