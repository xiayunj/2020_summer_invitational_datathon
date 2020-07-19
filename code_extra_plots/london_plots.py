import pandas as pd
import matplotlib.pyplot as plt
import joypy
import seaborn as sns
import os
import numpy as np
from os.path import join
# Set the GGPlot style
plt.style.use('ggplot')

# Loading the data
data = {"London": {}, "Rio": {}, "Vancouver": {}}
ROOT = "/Users/evanvogelbaum/CitadelDatathon/CitadelDatathonSummer2020/datasets_full"
paths = [os.path.join(ROOT, s) for s in data.keys()]
skip = ["infrastructure_spending", "tickets_for_sale"]  # datasets with odd encoding schemes
for path in paths:
    for csv in os.listdir(path):
        if not ".csv" in csv or any([s in csv for s in skip]): continue
        name = csv[:csv.find(".csv")]
        data[os.path.split(path)[1]][name] = pd.read_csv(os.path.join(path, csv), low_memory=False)
relevant_regions = {'East Midlands', 'East', 'London', 'North East', 'North West', 'Northern Ireland', 'Scotland',
 'South East', 'South West', 'Wales', 'West Midlands', 'Yorkshire and The Humber'}
earnings = data['London']['london_earnings_by_borough']
earnings = earnings[~(earnings.pay.isna() |  earnings.pay_type.isna())]
# Conversion of hourly pay to weekly pay by assuming a 40 hour work week
multiples = pd.Series([40.0 if pt == "Hourly" else 1.0 for pt in earnings.pay_type])
earnings.pay = pd.Series(np.array(earnings.pay) * np.array(multiples)).values
relevant_earnings = earnings[(earnings.area.isin(relevant_regions))]
# Make boxplots of relevant Regions
ax = sns.boxplot(data=relevant_earnings, x='year', y='pay', color='lightblue')
for item in ax.get_xticklabels():
    item.set_rotation(90)
title = "Boxplots of Distribution of Pay over the UK"
plt.title(title)
plt.ylabel("Pay (£)")
plt.savefig(title + ".png")
plt.clf()
# Filter any  outliers via IQR method
print("Before, relevant earnings is of size {}".format(len(relevant_earnings)))
for year, d in relevant_earnings.groupby("year"):
    print("Before, min, max are: ({}, {})".format(min(d.pay), max(d.pay)))
    Q1 = d.pay.quantile(0.25)
    Q3 = d.pay.quantile(0.75)
    IQR = Q3 - Q1
    print("Lower bound is {}, upper bound is {}".format(Q1 - 1.5 * IQR, Q3 + 1.4 * IQR))
    d.query('@Q1 - 1.5 * @IQR <= pay <= (@Q3 + 1.5 * @IQR)', inplace=True)
    print("After, min, max are: ({}, {})".format(min(d.pay), max(d.pay)))
print("After, relevant earnings is of size {}".format(len(relevant_earnings)))
# Show distribution of pay across relevant boroughs
for region, d in relevant_earnings.groupby("area"):
    labels = [y if y%2 == 0 else None for y in list(d.year.unique())]
    fig, axes = joypy.joyplot(d, by="year", column="pay", labels=labels,
                              range_style='own', grid="y", linewidth=1, legend=False, figsize=(6, 5))
    plt.xlabel("Pay (£)")
    title = "{} Distribution of Pay over Time".format(region); plt.title(title)
    plt.savefig(title + ".png", bbox_inches='tight')
    plt.clf()
# Showing the overall distribution of pay across all boroughs
labels = [y if y%2 == 0 else None for y in list(relevant_earnings.year.unique())]
fig, axes = joypy.joyplot(relevant_earnings, by="year", column="pay", labels=labels,
                              range_style='own', grid="y", linewidth=1, legend=False, figsize=(6, 5))
plt.xlabel("Pay (£)")
title = "UK Distribution of Pay over Time"; plt.title(title)
plt.savefig(title + ".png", bbox_inches='tight')

#Making Distribution of income by borough
for area in earnings['area'].unique():
    filtered = earnings[earnings['area'] == area]['pay']
    sns.distplot(filtered, bins=50)
    plt.title(area)
    plt.xlabel("Pay")
    plt.savefig(os.path.join(area + ".png"))
    plt.clf()
# Making the Joyplot of distirbution of earnings over time
labels = [y if y%2 == 0 else None for y in list(earnings.year.unique())]
fig, axes = joypy.joyplot(earnings, by="year", column="pay", labels=labels,
                          range_style='own', grid="y", linewidth=1, legend=False, figsize=(6, 5))
plt.xlabel("Pay")
plt.title("Distribution of pay over the years across all boroughs")
plt.show()
# Plotting the average pay over time across all boroughs
avg = [
    earnings[earnings.year == y]['pay'].mean() for y in sorted(earnings.year.unique())
]
labels = [y if int(y)%2==0 else "" for y in sorted(earnings.year.unique())]
plt.plot(avg)
plt.ylabel("Average Pay")
plt.xlabel("Years")
plt.xticks(range(len(avg)), labels)
plt.title("Average Earnings over time Across Boroughs")
plt.show()


# Some intial work on the economic_activity data
economic_activity = data["London"]["london_economic_activity"]
economic_activity = economic_activity.dropna()
economic_activity = economic_activity[~(economic_activity.area.isin(["Great Britain", "England"]))]

# Work on the travel data
visits = data['London']['UK_international-visits']
visits.area = visits.area.str.strip().apply(lambda x: x.replace("/", " or "))
visits.year = visits.year.replace('2019P', '2019')
print("Statistics on stays by area")
print(visits.groupby("area").visits.agg(["mean", "min", "max", "var", "size"]))
regional_visits = visits[~(visits.area.isin(["TOTAL ENGLAND", "ALL STAYING VISITS"]))]
relevant_regions = {"SCOTLAND", "NORTHERN IRELAND", "WALES", "NORTH EAST", "NORTH WEST", "YORKSHIRE", "WEST MIDLANDS",
"EAST MIDLANDS", "SOUTH WEST", "SOUTH EAST", "EAST OF ENGLAND", "LONDON"}
relevant_visits = regional_visits[(regional_visits.area.isin(relevant_regions))]

def quarter2number(x):
    if x == "January-March":
        return "Q1"
    elif x == "April-June":
        return "Q2"
    elif x == "July-September":
        return "Q3"
    elif x == "October-December":
        return "Q4"
    else:
        raise Exception("Inccorect input {}".format(x))

relevant_visits["year"] = relevant_visits.year + relevant_visits.quarter.apply(quarter2number)
# Plotting visits over time
sums = relevant_visits.groupby("year").visits.sum()
labels = [x[:-2] if "Q1" in x and int(x[:-2]) % 2 == 0 else None for x in sums.index]
plt.plot(sums, color="black");
plt.title("UK International Tourists from 2002 to 2019");
plt.ylabel("Tourists (thousands)");
plt.xticks(np.arange(len(sums)), labels);
plt.savefig("UK International Tourists Over Time.png", bbox_inches='tight')


## Plots of visitation data by region
pairs = [("SCOTLAND", "NORTHERN IRELAND"), ("WALES", "NORTH EAST"), ("NORTH WEST", "YORKSHIRE"), ("WEST MIDLANDS",
"EAST MIDLANDS"), ("SOUTH WEST", "SOUTH EAST"), ("EAST OF ENGLAND", "LONDON")]

def plot_subplot_column(ax, col, data):
    for i, value in enumerate(["visits", "spend", "nights"]):
        averages = data.groupby("year")[value].mean()
        labels = [x[:-2] if "Q1" in x and int(x[:-2]) % 3 == 0 else None for x in averages.index]
        ax[i, col].plot(averages);
        ax[i, col].set_title(value.title())
        ax[i, col].tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=True)
        if col==0:
            if value == "visits":
                ax[i, col].set_ylabel("Visits (thousands)", fontsize=8)
            elif value == "spend":
                ax[i, col].set_ylabel("£ Spent (millions)", fontsize=8)
            elif value == "nights":
                ax[i, col].set_ylabel("Nights Stayed (thousands)", fontsize=8)
            else:
                raise Exception("Bad value")
        if i == 2:
            ax[i, col].set_xticklabels(labels)
        else:
            ax[i, col].set_xticklabels((None,) * len(labels))

for place1, place2 in pairs:
    fig, ax = plt.subplots(3, 2)
    p1data = relevant_visits[relevant_visits.area == place1]
    p2data = relevant_visits[relevant_visits.area == place2]
    fig.suptitle("{} and {}".format(place1.title(), place2.title()))
    plot_subplot_column(ax, 0, p1data)
    plot_subplot_column(ax, 1, p2data)
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    plt.savefig("{} and {} tourist data.png".format(place1.title(), place2.title()))
    plt.clf()
plt.clf();plt.cla()
for column in ['visits', 'spend', 'nights']:
    plt.clf();plt.cla()
    fig = plt.gcf()
    fig.set_size_inches(15, 7)
    for region, data in relevant_visits.groupby("area"):
        yearly = data.groupby("year")[column].mean()
        plt.plot(yearly, label=region)
    labels = [x[:-2] if "Q1" in x and int(x[:-2]) % 3 == 0 else None for x in yearly.index]
    plt.xticks(np.arange(len(labels)), labels)
    title = "{} Over Time By Regions".format(column.title())
    if column == "visits":
        plt.ylabel("Visits (thousands)")
    elif column == "spend":
        plt.ylabel("£ Spent (millions)")
    elif column == "nights":
        plt.ylabel("Nights Stayed (thousands)")
    plt.legend(bbox_to_anchor=(1.01, 1));plt.title(title)
    plt.axvline(x=list(yearly.index).index("2012Q2"), c="red", linestyle='--', linewidth=4)
    plt.savefig(title + ".png", bbox_inches='tight')
