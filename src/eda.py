import matplotlib.pyplot as plt

def charges_distribution(df):

    fig, ax = plt.subplots()

    ax.hist(
        df['charges'],
        bins=20
    )

    ax.set_title(
        'Distribution of Charges'
    )

    ax.set_xlabel(
        'Charges'
    )

    ax.set_ylabel(
        'Frequency'
    )

    return fig


def bmi_vs_charges(df):

    fig, ax = plt.subplots()

    ax.scatter(
        df['bmi'],
        df['charges']
    )

    ax.set_title(
        'BMI vs Charges'
    )

    ax.set_xlabel(
        'BMI'
    )

    ax.set_ylabel(
        'Charges'
    )

    return fig


def age_vs_charges(df):

    fig, ax = plt.subplots()

    ax.scatter(
        df['age'],
        df['charges']
    )

    ax.set_title(
        'Age vs Charges'
    )

    ax.set_xlabel(
        'Age'
    )

    ax.set_ylabel(
        'Charges'
    )

    return fig