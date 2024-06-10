import numpy as np
import pandas as pd
import math
from itertools import chain, combinations
from tqdm import tqdm
import matplotlib.pyplot as plt


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def get_late_days(lateness):
    hh, mm, ss = lateness.split(':')
    return math.ceil(int(hh) / 24)


if __name__ == '__main__':
    assignments_df = pd.read_csv('input/EE_CNS_CS_148_Spring_2024_grades.csv')
    readings_df = pd.read_csv('input/Paper Readings - checks.csv')

    class_df = pd.read_csv('input/ClassRollSheet.csv')
    class_df = class_df[class_df['Type'] == 'Student']

    grades_df = pd.merge(class_df, assignments_df, left_on='Caltech UID', right_on='SID', how='left')
    grades_df = pd.merge(grades_df, readings_df, left_on='Caltech UID', right_on='Caltech UID', how='left')

    final_grades_df = grades_df[['Caltech UID', 'Name', 'Class', 'Option']]

    assignments = []
    extra_credit = {'HW1': 5, 'HW2': 11, 'HW3': 4, 'HW4': 10}

    hws = [1, 2, 3, 4]
    for hw in hws:
        assignments.append(f'HW{hw}')
        final_grades_df[f'HW{hw}_pts'] = grades_df[f'Homework {hw} - Code'].fillna(0) +\
                                         grades_df[f'Homework {hw} - Written Part'].fillna(0)
        final_grades_df[f'HW{hw}_raw_max'] = (grades_df[f'Homework {hw} - Code - Max Points'] +
                                          grades_df[f'Homework {hw} - Written Part - Max Points'])
        final_grades_df[f'HW{hw}_ec'] = extra_credit[f'HW{hw}']
        final_grades_df[f'HW{hw}_max'] = final_grades_df[f'HW{hw}_raw_max'] - final_grades_df[f'HW{hw}_ec']
        final_grades_df[f'HW{hw}_pct'] = final_grades_df[f'HW{hw}_pts'] / final_grades_df[f'HW{hw}_max']

        grades_df[f'Homework {hw} - Code - Late days'] = \
            grades_df[f'Homework {hw} - Code - Lateness (H:M:S)'].apply(get_late_days)
        grades_df[f'Homework {hw} - Written Part - Late days'] = \
            grades_df[f'Homework {hw} - Written Part - Lateness (H:M:S)'].apply(get_late_days)

        final_grades_df[f'HW{hw}_lateness'] = grades_df[[f'Homework {hw} - Code - Late days',
                                                            f'Homework {hw} - Written Part - Late days']].max(axis=1)

        final_grades_df[f'HW{hw}_weight'] = 0.2

    prs = [1, 2, 3, 4, 5, 7, 8]
    for pr in prs:
        assignments.append(f'PR{pr}')
        final_grades_df[f'PR{pr}_pts'] = grades_df[f'Week {pr} - Done'].fillna(0)
        final_grades_df[f'PR{pr}_pct'] = final_grades_df[f'PR{pr}_pts'] / 1.

        final_grades_df[f'PR{pr}_lateness'] = grades_df[f'Week {pr} - Lateness'].fillna(0)
        final_grades_df[f'PR{pr}_weight'] = 0.1 / len(prs)

    assignments.append('proposal')
    final_grades_df['proposal_pts'] = grades_df['Project Proposal'].fillna(0)
    final_grades_df['proposal_pct'] = final_grades_df['proposal_pts'] / grades_df['Project Proposal - Max Points']
    final_grades_df['proposal_lateness'] = 0
    final_grades_df['proposal_weight'] = 0.1

    final_grades_df['total_lateness'] = 0
    for assignment in assignments:
        final_grades_df[f'{assignment}_weighted'] = (final_grades_df[f'{assignment}_pct'] *
                                                     final_grades_df[f'{assignment}_weight'])
        final_grades_df['total_lateness'] += final_grades_df[f'{assignment}_lateness']

    final_grades_df['total'] = -np.nan
    final_grades_df['total_combination'] = ""
    final_grades_df['total_combination_lateness'] = np.nan
    max_lateness = 6

    assignments_powerset = list(powerset(assignments))
    for i, row in tqdm(final_grades_df.iterrows(), total=len(final_grades_df)):
        max_combination = -np.inf
        combination = ""
        combination_lateness = 0
        for subset in assignments_powerset:
            total = 0
            lateness = 0
            for assignment in subset:
                total += row[f'{assignment}_weighted']
                lateness += row[f'{assignment}_lateness']
            if lateness <= max_lateness and total > max_combination:
                max_combination = total
                combination = str(subset)
                combination_lateness = lateness

        final_grades_df.at[i, 'total'] = max_combination
        final_grades_df.at[i, 'total_combination'] = combination
        final_grades_df.at[i, 'total_combination_lateness'] = combination_lateness

    final_grades_df.to_csv('output/final_grades.csv', index=False)

    # plot histogram of final grades
    fig, ax = plt.subplots()
    ax.hist(final_grades_df['total'], bins=20, edgecolor='black')
    ax.set_xlabel('Final Grade')
    ax.set_ylabel('Number of Students')
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_title(f'Final Grades\nmean: {final_grades_df["total"].mean():.2f}, std: {final_grades_df["total"].std():.2f}')
    plt.savefig('output/final_grades_histogram.png')
    plt.show()
    plt.clf()

    fig, ax = plt.subplots()
    y = [0] + sorted(list(final_grades_df['total']))
    x = np.linspace(0, 1, len(y))
    ax.plot(x, y)
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.grid()
    ax.set_xlabel('Student percentile')
    ax.set_ylabel('Final Grade')
    ax.set_title('Final Grades CDF')
    plt.savefig('output/final_grades_cdf.png')
    plt.show()

    quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for quantile in quantiles:
        print(f'{quantile * 100:.0f}th percentile: {final_grades_df["total"].quantile(quantile):.2f}')


    for cutoff in np.arange(0.1, 1.1, 0.1):
        print(f"cutoff: {cutoff:.1f}: {len(final_grades_df[final_grades_df['total'] >= cutoff]) / len(final_grades_df):.2f}")

    for cutoff in np.arange(0.1, 1.1, 0.1):
        print(f"cutoff: {cutoff:.1f}: {len(final_grades_df[final_grades_df['total'] >= cutoff]) / len(final_grades_df):.2f}")





